import asyncio
import websockets
import json
import pickle

msg = \
{
  "jsonrpc" : "2.0",
  "id" : 8772,
  "method" : "public/get_order_book",
  "params" : {
    "instrument_name" : "BTC-PERPETUAL",
    "depth" : 5
  }
}

async def call_api(msg):
    global storage
    async with websockets.connect('wss://www.deribit.com/ws/api/v2') as websocket:
        await websocket.send(msg)
        while websocket.open:
            response = await websocket.recv()
            response = json.loads(response)['result']
            # do something with the response...
            storage.append(response)
            # if len(storage) == 10:
            with open("test.pkl", "wb") as fp:  # Pickling
                pickle.dump(response, fp)
            websocket.close()


asyncio.get_event_loop().run_until_complete(call_api(json.dumps(msg)))

