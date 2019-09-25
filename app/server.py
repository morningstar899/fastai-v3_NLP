from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import StringIO

from fastai import *
from fastai.text import *

export_file_url = 'https://www.googleapis.com/drive/v3/files/1DVEO3fbsnpEoDnquEhbn1Y9O6Pd36NjQ?alt=media&key=AIzaSyC71u1AJ2fmBoqpLTc7R34jtP2Il-L7ySQ'
#export_file_url = 'https://www.googleapis.com/drive/v3/files/1xUku3breeGz1-f3MlmlFwq58pbJoLuVR?alt=media&key=AIzaSyBVEqpZp8wHfzX7a7k9BM1vYaqwO68IiQo'
#export_file_url = 'https://drive.google.com/uc?export=download&id=1uteRlmYYOp0QxSAWYzFgfzT8WQnxP-iw'
export_file_name = 'fine_tuned.pkl'
#export_file_name = 'fine_tuned_JOB_DUTIES.pkl'
#export_file_name = 'fine_tuned_eJOB_ads.pkl'
classes = ['negative', 'positive']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

# PSEUDOCODE:
# func download_file
# Check to see if destination of to-be-downloaded file exists
# if it does, communicate with server and wait for a response
# Check to see if specified .pkl (extension) file exists in the directory
# if file does not exist, download it at specified destination
# read data from specified .pkl file
# 
# func setup_learner
# if download_file worked, load up the .pkl file containing weights and activations
### Note: In this case weights and activations are being used for the computer to know 
### how important a combination of words is. Ex: It was a hot _ . 
### "It was a hot 'day'" would rank higher and thus have a higher weight 
### than "It was a hot 'highly'"
# if download_file did not work, return one of two errors
# 1) if model (aforementioned learner) was trained using a CPU, return a specific error
# 2) all other errors, return a generic error


async def download_file(url, dest):
    # early exit if destination exists
    if dest.exists(): return
    
    # section for downloading the file. In this case will be used to download the weights (.pkl) 
    # for the model
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    
        # used to load the learner that will be used. In this case, will call on fastai's text learner
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.json()
    content = data['textField']

    prediction = learn.predict(content, 60, temperature=0.10)
    #prediction = print("\n".join(learn.predict(content, 90, temperature=0.75) for _ in range(1)))
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
