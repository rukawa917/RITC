import requests, json
import pandas as pd
import numpy as np
import time
import threading
import scipy.stats as si
from py_vollib.black_scholes import implied_volatility as bs

pd.set_option('display.max_columns', 500)


class Client():
    '''
    Basic api call methods
    '''
    base = "http://localhost:9999/v1"

    def __init__(self, apikey):
        self.client = requests.Session()
        self.client.headers.update({"X-API-Key": apikey})

        self.res = self.client.get(self.base + '/limits')

        if self.res.ok:
            # For some reason it returns a list rather than single dict
            limits = self.res.json()[1]
            self.gross_limit = limits['gross_limit']
            self.net_limit = limits['net_limit']
            print("[OptionsExecManager] Options Position Net Limits: %s" % self.net_limit)
        else:
            print('[Execution Manager] Error could not obtain position limits from API!')

    def get_case(self):
        resp = self.client.get(f'{self.base}/case').json()
        return resp

    def get_trader(self):
        resp = self.client.get(f'{self.base}/trader').json()
        return resp

    def get_limits(self):
        resp = self.client.get(f'{self.base}/limits').json()
        df = pd.DataFrame.from_records(resp)
        return df

    def get_news(self):
        resp = self.client.get(f'{self.base}/news')
        return resp

    def get_assets(self):
        resp = self.client.get(f'{self.base}/assets').json()
        df = pd.DataFrame.from_records(resp)
        return df

    def get_securities(self):
        resp = self.client.get(f'{self.base}/securities').json()
        df = pd.DataFrame.from_records(resp)
        return df

    def get_securities_ob(self, param):
        '''
        :param param: dict
            :param keys: ticker(required): string, limit: int
        '''
        resp = self.client.get(f'{self.base}/securities/book', params=param).json()
        df_bid = pd.DataFrame.from_records(resp['bids'])
        df_ask = pd.DataFrame.from_records(resp['asks'])
        return df_bid, df_ask

    def get_securities_hist(self, param):
        '''
        :param param: dict
            :param keys: ticker(required): string, period:int, limit:int
        '''
        resp = self.client.get(f'{self.base}/securities/history', params=param).json()
        df = pd.DataFrame.from_records(resp)
        return df

    def get_securities_tas(self, param):
        '''
        :param param: dict
            :param keys: ticker(required): string, after:int, period:int, limit:int
        '''
        resp = self.client.get(f'{self.base}/securities/tas', params=param).json()
        df = pd.DataFrame.from_records(resp)
        return df

    def get_order(self, param):  # user order
        resp = self.client.get(f'{self.base}/orders', params=param).json()
        df = pd.DataFrame.from_records(resp)
        return df

    def post_order(self, param):
        '''
        :param keys: ticker(required), type-LIMIT/MARKET(required), quantity(required), action-BUY/SELL(required),
                        price(when limit, required)
                        dry_run(only available when type is market, simulates order execution)
        '''
        resp = self.client.post(f'{self.base}/orders', params=param)
        if resp.ok:
            return resp.json()
        elif resp.status_code == 429:
            # print(f"api limit reached, waiting for {resp.json()['wait']}s")
            time.sleep(resp.json()['wait'])
            resp = self.client.post(f'{self.base}/orders', params=param)
            return resp.json()

    def get_order_id(self, id):
        '''
        :param keys: id(required): int
        '''
        resp = self.client.get(f'{self.base}/orders/{id}').json()
        return resp

    def get_active_tender(self):
        resp = self.client.get(f'{self.base}/tenders').json()
        return resp

    def accept_tender_offer(self, id):
        resp = self.client.post(f'{self.base}/tenders/{id}').json()
        return resp

    def decline_tender_offer(self, id):
        resp = self.client.delete(f'{self.base}/tenders/{id}').json()
        return resp

    def vol_forecast(self):
        news = self.get_news()
        # news = requests.get(self.base + '/news', params={'limit': 1}, headers=self.headers)
        if news.ok:
            body = news.json()[0]['body']  # call the body of the news article

            if body[4] == 'l':  # 'the latest annualised' - direct figure
                sigma = int(body[-3:-1]) / 100
            elif body[4] == 'a':  # 'the annualized' - expectation of range
                sigma = (int(body[-26:-24]) + int(body[-32:-30])) / 200
            # sigma = int(body[-26:-24]) / 100
            else:
                sigma = 0.2
        else:
            print('[Indicators] Could not reach API! %s' % self.res.json())
            sigma = 0.2
        # print("Vol Forecast is", sigma)
        return sigma

    def delete_order_id(self, id):
        resp = self.client.delete(f'{self.base}/orders/{id}').json()
        return resp

    def bulk_delete_order(self, param):
        '''
        Bulk cancel open orders. Exactly one query parameter must be specified. If multiple query parameters are
        specified, only the first available parameter in the order below will be processed. Returns a result object that
        specifies which orders were actually cancelled.

        :param keys:
            all: Set to 1 to cancel all open orders
            ticker: Cancel all open orders for a security.
            ids: Cancel a set of orders referenced via a comma-separated list of order ids. For example, 12,13,91,1.
            query: Query string to select orders for cancellation. For example, Ticker='CL' AND Price>124.23 AND
                    Volume<0 will cancel all open sell orders for CL priced above 124.23.
        :return:
        '''
        resp = self.client.post(f'{self.base}/commands/cancel', params=param).json()
        return resp


######################Strategies################################
class trader(Client):
    '''
    Class that uses basic api to do further analysis/execution
    '''
    def __init__(self, apikey, order):
        super().__init__(apikey)
        self.order_size = order

    def compute_delta(self, S, K, T, r, sigma, option):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        result = 0
        if option == 'call' or option == 'C':
            result = si.norm.cdf(d1, 0.0, 1.0)
        if option == 'put' or option == 'P':
            result = -si.norm.cdf(-d1, 0.0, 1.0)
        return result

    def compute_pf_delta_change(self, inv, K, T, r):  # announcement sigma
        '''
        :param inv: positions
        :param K: Strike
        :param T: time left till expiry
        :param r: interest rate = 0
        :return: # of shares required to make delta = 0
        '''
        inv['Delta'] = 0
        for i in range(1, 41):  # go over options inventory
            position = inv.loc[i, 'position']
            inventory = self.get_securities()
            S = inventory['last'][0]

            try:
                cb_iv = bs.implied_volatility(order_book.loc[29, 'bid'], S, K, T, r, 'c')
                ca_iv = bs.implied_volatility(order_book.loc[29, 'ask'], S, K, T, r, 'c')
            except:
                cb_iv = 0
                ca_iv = 0

            if cb_iv == 0:
                print('The volatility is below the intrinsic value.')
                return 0
            # sigma = (1 + (underlying - strike) ** 3 / underlying) * sigma

            # compute_delta(self, S, K, T, r, sigma, option)
            if position < 0:  # short option position -> buy
                d = self.compute_delta(S, K, T, r, ca_iv, 'C') * inventory['position'][i]
                inv.at[i, 'Delta'] = d
            elif position > 0:  # long option position -> sell
                d = self.compute_delta(S, K, T, r, cb_iv, 'C') * inventory['position'][i]
                inv.at[i, 'Delta'] = d

        print(f"delta change: {round(-inv.at[0, 'position'] - inv['Delta'].sum() * 100, 0)}")
        # d_before - d_now       spot before      spot now
        # 70 - 50 = 20              -70             -50      Buy 20
        # 100 - -50 = 150           -100            50        Buy 150
        # -100 - -20 = -80           100            20        Sell 80
        # return how many underlying we should trade
        return round(-inv.at[0, 'position'] - inv['Delta'].sum() * 100, 0)
        # return round(inv['Delta'].sum() * 100 + inv.at[0, 'position'], 0)

    def exit_all(self):  # exit all the remaining positions in underlying and options
        inventory = self.get_securities()
        for i in range(0, len(inventory)):
            ticker = inventory['ticker'][i]
            position = int(inventory['position'][i])
            if position > 0:
                # print(f"{ticker}: {position}")
                if position > 10000:
                    count = int(position / 10000)
                    remain = position
                    for i in range(count + 1):
                        print(f"remain: {remain}")
                        if remain >= 10000:
                            param = {'ticker': ticker, 'type': "MARKET",
                                     'quantity': 10000, 'action': 'SELL'}
                            print(self.post_order(param))
                            remain -= 10000
                        elif remain == 0:
                            break
                        else:
                            param = {'ticker': ticker, 'type': "MARKET",
                                     'quantity': remain, 'action': 'SELL'}
                            print(self.post_order(param))
                    return param

                else:
                    param = {'ticker': ticker, 'type': "MARKET",
                             'quantity': abs(position), 'action': 'SELL'}
                    print(self.post_order(param))
                    return param
            elif position < 0:
                if position < -10000:
                    count = int(position / 10000)
                    remain = position
                    param = {}
                    for i in range(count + 1):
                        print(f"remaining: {remain}")
                        if remain <= -10000:
                            param = {'ticker': ticker, 'type': "MARKET",
                                     'quantity': 10000, 'action': 'BUY'}
                            print(self.post_order(param))
                            remain += 10000
                        else:
                            param = {'ticker': ticker, 'type': "MARKET",
                                     'quantity': -remain, 'action': 'SELL'}
                            print(self.post_order(param))
                    return param
                else:
                    param = {'ticker': ticker, 'type': "MARKET",
                             'quantity': abs(position), 'action': 'BUY'}
                    print(self.post_order(param))
                    return param


############################################################################################
    def enter_mp_op(self, K, T, r, upper_bound, lower_bound):  # enter mispriced options
        '''
        monitor the market IV, if the IV is too high or low, enter

        :param upper_bound: upper bound of fair volatility
        :param lower_bound: lower bound of fair volatility
        :return: None
        '''
        time.sleep(0.2)
        ob = self.get_securities()

        # compute implied/market volatility
        S = ob['last'][0]
        try:
            cb_iv = bs.implied_volatility(order_book.loc[29, 'bid'], S, K, T, r, 'c')  # call bid iv
            ca_iv = bs.implied_volatility(order_book.loc[29, 'ask'], S, K, T, r, 'c')
        except:
            cb_iv = 0
            ca_iv = 0

        if cb_iv == 0:
            print('The volatility is below the intrinsic value.')
            return

        # (S, K, T, ob.loc[29, 'bid'], r, sigma_pred, option='C')
        print(f"market call bid iv: {cb_iv} vs fair_upper_v: {upper_bound}")
        print(f"market call ask iv: {ca_iv} vs fair_lower_v: {lower_bound}")

        # if iv is higher than upper bound of theoretical vol, enter short vol (sell option, buy underlying)
        if cb_iv > upper_bound:
            print(f"option overpriced")
            param1 = {'ticker': ticker, 'type': "MARKET",
                      'quantity': 1, 'action': 'SELL'}  # short call imp vol too high
            t1 = threading.Thread(target=self.post_order, args=(param1,))

            # delta hedge
            delta_hedge_amt = self.compute_delta(S, K, T, r, cb_iv, 'call')  # delta hedge with call bid iv
            param2 = {'ticker': 'RTM', 'type': "MARKET",
                      'quantity': delta_hedge_amt * 100, 'action': 'BUY'}
            t2 = threading.Thread(target=self.post_order, args=(param2,))
            print(f"option: {param1}\nunderlying: {param2}")

            t1.start()
            t2.start()
            t1.join()
            t2.join()
        # if iv is lower than lower bound of theoretical vol, enter long vol (buy option, sell underlying)
        elif ca_iv < lower_bound:
            print(f"option underpriced\nmarket_iv:{ca_iv} vs fair_iv:{lower_bound}")
            param1 = {'ticker': ticker, 'type': "MARKET",
                      'quantity': 1, 'action': 'BUY'}  # long call imp vol too low
            t1 = threading.Thread(target=self.post_order, args=(param1,))

            delta_hedge_amt = self.compute_delta(S, K, T, r, ca_iv, 'call')
            param2 = {'ticker': 'RTM', 'type': "MARKET",
                      'quantity': delta_hedge_amt * 100, 'action': 'SELL'}
            t2 = threading.Thread(target=self.post_order, args=(param2,))
            print(f"option: {param1}\nunderlying: {param2}")

            t1.start()
            t2.start()
            t1.join()
            t2.join()


    def monitor_fair_vol(self, K, T, r, upper_bound, lower_bound):
        '''
        monitor IV in the market and exit if market IV is fair vol

        :param upper_bound: upper bound of fair volatility
        :param lower_bound: lower bound of fair volatility
        :return: None
        '''
        time.sleep(0.2)
        order_book = self.get_securities()
        S = order_book.loc[0, 'last']  # last price of RTM

        # compute implied vol
        try:
            cb_iv = bs.implied_volatility(order_book.loc[29, 'bid'], S, K, T, r, 'c')
            ca_iv = bs.implied_volatility(order_book.loc[29, 'ask'], S, K, T, r, 'c')
        except:
            cb_iv = 0
            ca_iv = 0

        if cb_iv == 0:
            print('The volatility is below the intrinsic value.')
            return

        # if call bid iv is within the range of upper and lower bound, we exit all pos
        if (cb_iv < upper_bound) and (cb_iv > lower_bound):
            if order_book.loc[29, 'position'] < 0:
                print("fairly priced! exit!!!!")
                param1 = {'ticker': ticker, 'type': "MARKET",
                          'quantity': abs(order_book.loc[29, 'position']),
                          'action': 'BUY'}  # short call imp vol too high
                param2 = {'ticker': 'RTM', 'type': "MARKET",
                          'quantity': order_book.loc[0, 'position'],
                          'action': 'SELL'}  # short call imp vol too high

                t1 = threading.Thread(target=self.post_order, args=(param1,))
                t2 = threading.Thread(target=self.post_order, args=(param2,))

                t1.start()
                t2.start()
                t1.join()
                t2.join()
                print(param1)
                print(param2)

        elif (ca_iv < upper_bound) and (ca_iv > lower_bound):
            if order_book.loc[29, 'position'] > 0:
                print("fairly priced! exit!!!!")
                param1 = {'ticker': ticker, 'type': "MARKET",
                          'quantity': order_book.loc[29, 'position'],
                          'action': 'SELL'}  # short call imp vol too high
                param2 = {'ticker': 'RTM', 'type': "MARKET",
                          'quantity': abs(order_book.loc[0, 'position']),
                          'action': 'BUY'}  # short call imp vol too high

                t1 = threading.Thread(target=self.post_order, args=(param1,))
                t2 = threading.Thread(target=self.post_order, args=(param2,))

                t1.start()
                t2.start()
                t1.join()
                t2.join()


    def pf_delta_rebal(self, K, T, r):
        '''
        execution of portfolio delta rebalancing
        '''
        time.sleep(0.2)
        print()
        inv = self.get_securities()
        # compute delta change in portfolio
        if inv.loc[0, 'position'] < 0:
            order_amt = self.compute_pf_delta_change(inv, K, T, r)
        elif inv.loc[0, 'position'] > 0:
            order_amt = self.compute_pf_delta_change(inv, K, T, r)
        else:
            return

        # if change is larger or smaller than 5% of total position, rebalance
        if order_amt > int(inv.loc[0, 'position'] * 0.05):
            param = {'ticker': 'RTM', 'type': "MARKET",
                     'quantity': order_amt, 'action': 'BUY'}

            print(self.post_order(param))

        elif order_amt < int(inv.loc[0, 'position'] * 0.05) * (-1):
            param = {'ticker': 'RTM', 'type': "MARKET",
                     'quantity': abs(order_amt), 'action': 'SELL'}

            print(self.post_order(param))


    def compute_realized_vol(self):
        '''
        1. calculate historical volatility / hist_vol
        2. when the news is out, compare the avg sigma and hist / avg_sig
        3. if hist_vol < avg_sig -> vol increase  vice versa when hist_vol > avg_sig
            eg) hist_vol = 15, avg_sg = 25 -> future vol = x = 25 * 2 - 15 = 35
            eg) hist_vol = 25, avg_sg = 15 -> future vol = x = 15 * 2 - 25 = 5

        :return: sigma_prediction
        '''
        param = {'ticker': 'RTM', 'limit': 15}
        df = self.get_securities_hist(param)
        tick = df.loc[0, 'tick']
        vol_p = np.log(df['close'] / df['close'].shift(1)).std() * np.sqrt(3780)
        print(f"HV: {round(vol_p, 4)}, tick: {tick}")
        # print(f"HVC: {round(vol_p, 4)}, HVC: {round(vol_p, 4)},tick: {tick}")
        return round(vol_p, 4), tick


###############################################################################
trader1 = trader('demo1', 1)
trades_num = 0

# 5. loop 1-4 until Announcement comes out
while trader1.get_case()['status'] == "ACTIVE":  # cant start very well
    news = trader1.get_news().json()
    order_book = trader1.get_securities()
    tick = trader1.get_case()['tick']
    sigma_now = 0
    w = 1
    if news[0]['headline'][0] == 'N':
        fut_upper_bound = int(news[0]['body'][-26:-24]) / 100
        fut_lower_bound = int(news[0]['body'][-32:-30]) / 100
        if w == 1:
            sigma_now = 0.2
            w += 1
        else:
            sigma_now = int(news[1]['body'][-3:-1]) / 100
    elif news[0]['headline'][0] == 'A':
        sigma_now = int(news[0]['body'][-3:-1]) / 100
    else:
        sigma_now = 0.2

    call_bid = order_book.loc[29, 'bid']
    call_ask = order_book.loc[29, 'ask']
    ticker = order_book.loc[29, 'ticker']  # RM2C49
    S = order_book.loc[0, 'last']
    K = int(ticker[-2:])
    T = (2 * 300 - tick) / 300 / 12
    r = 0

    # entry_upper_bound = sigma_now + 0.02
    # entry_lower_bound = sigma_now - 0.05
    # exit_upper_bound = sigma_now + 0.01
    # exit_lower_bound = sigma_now - 0.01
    exit_upper_bound = (1 - (S - K) ** 2 / S) * sigma_now
    exit_lower_bound = (1 + (S - K) ** 2 / S) * sigma_now
    entry_upper_bound = (1 + (S - K) ** 3 / S) * sigma_now
    entry_lower_bound = (1 - (S - K) ** 3 / S) * sigma_now
    sigma_mid = (entry_lower_bound + entry_upper_bound) / 2

    if order_book.loc[29, 'position'] < 2 and order_book.loc[29, 'position'] > -2:
        print()
        trader1.enter_mp_op(K, T, r, entry_upper_bound, entry_lower_bound)
    trader1.pf_delta_rebal(K, T, r)
    trader1.monitor_fair_vol(K, T, r, exit_upper_bound, exit_lower_bound)

    # due to malfunction of exit_all() method, implemented simple version.
    # Since the vol value is set to different value at the start of every week,
    # exit all existing positions right before next week
    if tick % 75 > 73:
        print('\n\n')
        time.sleep(0.2)
        inv = trader1.get_securities()
        oppos = inv.loc[29, 'position']
        rtmpos = inv.loc[0, 'position']
        if oppos > 0:
            param = {'ticker': 'RTM2C49', 'type': "MARKET",
                     'quantity': oppos, 'action': 'SELL'}  # long call imp vol too low
            print(trader1.post_order(param))
            param = {'ticker': 'RTM', 'type': "MARKET",
                     'quantity': abs(rtmpos), 'action': 'BUY'}  # long call imp vol too low
            print(trader1.post_order(param))
        elif oppos < 0:
            param = {'ticker': 'RTM2C49', 'type': "MARKET",
                     'quantity': abs(oppos), 'action': 'BUY'}  # long call imp vol too low
            print(trader1.post_order(param))
            param = {'ticker': 'RTM', 'type': "MARKET",
                     'quantity': rtmpos, 'action': 'SELL'}  # long call imp vol too low
            print(trader1.post_order(param))
        quit()
        print("exit all")
