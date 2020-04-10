import os
import sys
sys.path.append(os.path.join(os.environ['dev'], 'CVLoc'))
import numpy as np
import h5py
import pandas as pd
try:
    import folium
except:
    print("Warning not folium package")

def getDataFrame(config):
    csvFile = os.path.join(os.environ['dev'], 'CVLoc', 'data',config['dataset'] + '.csv')
    names = ["pano_id", "yaw", "lat", "lon", "city", "neighbor", "bearing", "index"]
    frame = pd.read_csv(csvFile, names=names)
    return frame

def getBaseMap(config, frame, startIdx=None, zoom_start=17):
    startPoint = [frame.loc[startIdx-1, 'lat'], frame.loc[startIdx-1,'lon']]
    mymap = folium.Map(location=startPoint, zoom_start=zoom_start)
    return mymap

def getGtRoute(config, route):
    resultsFile = os.path.join( os.environ['dev'],'route-finder/results/ES', config['model'], config['zoom'], config['dataset'], config['fileName'])
    f = h5py.File(resultsFile, 'r')
    data = f.get('test_route')
    testRoutes = data[()].transpose()
    gtRoute = testRoutes[route, 0:config['m']]
    return gtRoute

# def getRoute(config, k=0):
#     resultsFile = os.path.join( os.environ['dev'], 'route-finder/results/ES', 
#                 config['model'], config['zoom'], config['dataset'], config['fileName'])
#     f = h5py.File(resultsFile, 'r')
#     data = f.get('best_estimated_top5_routes') # (500,1)
#     route_array = data[route][0] 
#     data = f.get(route_array) #(40,1)
#     top5points = data[m][0]
#     points = f.get(top5points)
#     points = points[()] # (40,5)
#     points = points.transpose()
#     route = points[k,:]
#     return route


def getBest5EstimatedMapPoints(config, route, m, k):
    if config['dataset'] == 'cmu5k':
        resultsFile = os.path.join( os.environ['dev'],'route-finder/results/ES', config['model'], config['zoom'], config['dataset'] + '_' + config['subset'], config['fileName'])
    else:
        resultsFile = os.path.join( os.environ['dev'],'route-finder/results/ES', config['model'], config['zoom'], config['dataset'], config['fileName'])
    
    f = h5py.File(resultsFile, 'r')
    data = f.get('best_estimated_top5_routes') # (500,1)
    route_array = data[route][0] 
    data = f.get(route_array) #(40,1) The 40 posible steps
    top5points = data[m][0] # Select step 
    points = f.get(top5points) # shape is (m,5) 
    points = points[()] 
    points = points.transpose() # Now shape is (5,m) 
    points = points[0:k,-1].reshape(-1) # Now return only the top k latest points
    return points


def getBest5EstimatedRoutes(config, route, m):
    if config['dataset'] == 'cmu5k':
        resultsFile = os.path.join( os.environ['dev'],'route-finder/results/ES', config['model'], config['zoom'], config['dataset'] + '_' + config['subset'], config['fileName'])
    else:
        resultsFile = os.path.join( os.environ['dev'],'route-finder/results/ES', config['model'], config['zoom'], config['dataset'], config['fileName'])
    
    f = h5py.File(resultsFile, 'r')
    data = f.get('best_estimated_top5_routes') # (500,1)
    route_array = data[route][0] 
    data = f.get(route_array) #(40,1) The 40 posible steps
    top5points = data[m-1][0] # Select step 
    points = f.get(top5points) # shape is (m,5) 
    points = points[()] 
    points = points.transpose() # Now shape is (5,m) 
    return points

def getPointRanking(config):
    import scipy.io as sio 
    
    if config['dataset'] == "cmu5k":
        fileName = os.path.join( os.environ['dev'],'route-finder/results/ES', config['model'], config['zoom'], config['dataset'] + '_'+ config['subset'], 'pointRanking.mat')
    else:
        fileName = os.path.join( os.environ['dev'],'route-finder/results/ES', config['model'], config['zoom'], config['dataset'], 'pointRanking.mat')
    
    #routesFile = os.path.join( 'temp/generated_routes.mat')
    ranking = sio.loadmat(fileName)['pointRanking']
    return ranking

def getRankingMatrix(config):
    if config['dataset'] == 'cmu5k':
        resultsFile = os.path.join( os.environ['dev'],'route-finder/results/ES', config['model'], config['zoom'], config['dataset'] + '_' + config['subset'], config['fileName'])
    else:
        resultsFile = os.path.join( os.environ['dev'],'route-finder/results/ES', config['model'], config['zoom'], config['dataset'], config['fileName'])

    f = h5py.File(resultsFile, 'r')
    data = f.get('ranking')
    testRoutes = data[()].transpose()
    ranking = testRoutes[:, 0:config['m']]
    return ranking

def getRoutesMatrix(config):
    import scipy.io as sio 

    if config['dataset'] == "cmu5k":
        routesFile = os.path.join( os.environ['dev'],'route-finder/Localisation/test_routes', config['dataset'] + '_routes_500_' + str(60) + '_' + config['subset'] + '.mat')
    else:
        routesFile = os.path.join( os.environ['dev'],'route-finder/Localisation/test_routes', config['dataset'] + '_routes_500_' + str(60) + '.mat')
    #routesFile = os.path.join( 'temp/generated_routes.mat')
    routes = sio.loadmat(routesFile)['test_route']
    return routes[:, 0:config['m']]

def getTurnMatrix(config):
    import scipy.io as sio 
    turnsFile = os.path.join( os.environ['dev'],'route-finder/Localisation/test_routes', config['dataset'] + '_turns_500_'+ str(config['turn_threshold']) + '.mat')
    turns = sio.loadmat(turnsFile)['test_turn']
    return turns[:,0:config['m']]

def getHighwayFlags(config):
    import scipy.io as sio
    filePath = os.path.join(os.environ['dev'], 'route-finder', 'Data', 'streetlearn', config['dataset'] + '_highwayflags.mat')
    flags = sio.loadmat(filePath)['highway_flag'].reshape(-1)
    return flags

def route2map(mymap, route, frame, color, routeIdx):
    for i,idx in enumerate(route):
        oidx = frame.loc[idx-1,'index']
        yaw = frame.loc[idx-1,'yaw']
        popup =  'Matlab Indices \n ->  Ridx: {}, m: {}, idx: {}, oidx: {}, gsvyaw: {}'.format(routeIdx + 1, i,idx,oidx, yaw)
        point = [frame.loc[idx-1,'lat'], frame.loc[idx-1,'lon']]
        folium.CircleMarker(
            location=point,
            radius=5,
            popup=popup,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(mymap)
    return mymap 