import os
import json
import glob
import numpy as np

def get_all_video(filename):
    videos_list = set()
    fp = open(filename)
    lines = fp.readlines()
    for line in lines:
        sp = line.rstrip("\n").split(" ")
        videoname = sp[0]
        videos_list.add(videoname)
    return list(videos_list)

def parse(filename, args):
    results = {}
    videos_list = get_all_video(filename)
    for videoname in videos_list:
        annofile = os.path.join(args.anno_path, videoname + ".json")
        with open(annofile, 'r') as f:
            data = json.load(f)
            num_mId = len(data['shapes']) - 1 # no need zones
            num_vId = 4 # 4 type of vehicles
        results[videoname] = np.zeros((args.K, num_mId, num_vId))
    fp = open(filename)
    lines = fp.readlines()
    for line in lines:
        sp = line.rstrip('\n').split()
        videoname = sp[0]
        assert videoname in results
        fId = int(sp[1])
        if fId > args.nframes:
            continue
        segId = (fId - 1) * args.K // args.nframes
        mId = int(sp[2]) - 1
        vId = int(sp[3]) - 1
        results[videoname][segId, mId, vId] += 1
    return results
 
def eval_video(args):
    pd = parse(args.count_outputs, args)
    gt = parse(args.tuning_submission, args)    
    diff = np.sum((gt[args.videoname] - pd[args.videoname])**2)
    scale = np.prod(gt[args.videoname].shape)
    return diff, scale

# def my_eval(TRACK_OUTPUTS, ANNO_FILES='anno_paths', PREDICTION='prediction.txt',
#             SUBMISSION='submission.txt', k=4, nframes=600, cam_tune=None, BEST=None, EVAL=False):    
#     if not EVAL:
#         return
#     pd = my_eval_parse(PREDICTION, TRACK_OUTPUTS, ANNO_FILES, k, nframes)
#     gt = my_eval_parse(SUBMISSION, TRACK_OUTPUTS, ANNO_FILES, k, nframes)
#     X, Y = 0, 0
    
#     # get videos name evaluate
#     #files = sorted(glob.glob(f"{TRACK_OUTPUTS}/cam*txt"))
#     #names = [os.path.basename(x).replace('.txt', '') for x in files]
#     names = sorted(list(gt.keys()))
    
    
#     if cam_tune and BEST:
#         names = [cam_tune]
        
#     for videoname in names:
#         diff = gt[videoname] - pd[videoname]
#         rmse = (diff**2).sum()
#         x = rmse
#         y = np.prod(gt[videoname].shape)
        
#         X += rmse
#         Y += np.prod(gt[videoname].shape)    
        
#         if BEST:
#             if x/y < BEST[videoname]:
#                 print("Better", videoname, BEST[videoname], '-->', x/y)
#             elif x/y > BEST[videoname]:
#                 print("Worse", videoname, BEST[videoname], '-->', x/y)
#             else:
#                 print('=')
            
        
#     res = np.sqrt(X / Y)
#     return res 

# def parse(filename):
#     results = {}
#     files = sorted(glob.glob(os.path.join(f'{TRACK_OUTPUTS}', 'cam*txt')))
#     names = [os.path.basename(x).replace('.txt', '') for x in files]
#     for x in names:
#         with open(f"{ANNO_FILES}/{x}.json", 'r') as f:
#             data = json.load(f)
#         total_mid = len(data['shapes']) - 1
#         results[x] = np.zeros((600, total_mid, 4))
#     fp = open(filename)
#     lines = fp.readlines()
#     for line in lines:
#         sp = line.rstrip("\n").split(" ")
#         videoname = sp[0]
#         assert videoname in results
#         fId = int(sp[1]) - 1 
#         mId = int(sp[2]) - 1
#         vId = int(sp[3]) - 1
#         results[videoname][fId, mId, vId] += 1
#     return results

# def compute_nwRMSE(n, pdArray, gtArray):
#     # weight
#     wVect = np.asarray(np.arange(1, n+1)) / (n * (n + 1) / 2.0)
#     fNum, mNum, typeNum = pdArray.shape
#     lst = range(0, fNum)
#     interval = int(math.ceil(fNum / float(n)))
#     segLsts = [lst[i : i + interval] for i in range(0, len(lst), interval)]
#     gtCntArray = np.zeros(mNum)
#     pdCntArray = np.zeros(mNum)
#     nwRMSEArray = np.zeros((mNum, typeNum))
#     wRMSEArray = np.zeros((mNum, typeNum))
#     vehicleNumArray = np.zeros((mNum, typeNum))
#     for mId in range(0, mNum):
#         gtCntArray[mId] = np.sum(gtArray[:, mId, :])
#         pdCntArray[mId] = np.sum(pdArray[:, mId, :])
#         for tId in range(0, typeNum):
#             # wRMSE
#             diffVectCul = np.zeros(n)
#             for segId, frames in enumerate(segLsts):
#                 diff = np.square(sum(pdArray[0:frames[-1], mId, tId]) - sum(gtArray[0:frames[-1], mId, tId]))
#                 diffVectCul[segId] = diff
#             wRMSE = np.sqrt(np.dot(wVect, diffVectCul))

#             # num
#             vehicleNum = np.sum(gtArray[:, mId, tId])
#             vehicleNumArray[mId, tId] = vehicleNum 

#             # for print only
#             if vehicleNum == 0:
#                 wRMSEArray[mId, tId] = 0 
#             else:
#                 wRMSEArray[mId, tId] = wRMSE / vehicleNum

#             #nwRMSE
#             if wRMSE > vehicleNum:
#                 nwRMSE = 0
#             else:
#                 if vehicleNum == 0:
#                     nwRMSE = 0
#                 else:
#                     nwRMSE = 1 - wRMSE / vehicleNum
#             ########################
#             #my thing
#             #nwRMSE = wRMSE / vehicleNum
#             ########################
#             nwRMSEArray[mId, tId] = nwRMSE
#     nwRMSEArray = np.multiply(nwRMSEArray, vehicleNumArray)
#     return np.sum(nwRMSEArray), np.sum(vehicleNumArray)

# def eval(k=4):
#     pd = parse(PREDICTION)
#     gt = parse(SUBMISSION)
#     X = 0
#     Y = 0
#     files = sorted(glob.glob(f"{TRACK_OUTPUTS}/cam*txt"))
#     names = [os.path.basename(x).replace('.txt', '') for x in files]
#     for videoname in names:
#         x, y = compute_nwRMSE(k, pd[videoname], gt[videoname])
#         X += x
#         Y += y
#     print('eval:', X/Y)