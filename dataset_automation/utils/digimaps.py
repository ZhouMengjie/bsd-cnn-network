import os 
import sys 
from shutil import copy2, rmtree

def splitGeometriesShapes(dataset):
    # Given a directory containing point, line and polygon shapefiles. This function 
    # splits the content according to the geometry type.
    datasetDir = os.path.join(os.environ['datasets'], 'Digimap', dataset, 'st')
    if os.path.isdir(datasetDir):
        print("Spliting geometries in {}".format(datasetDir))    
        # Prepare a directory where to save results 
        saveDir = os.path.join(os.environ['datasets'], 'Digimap', dataset, 'splitted')
        if os.path.isdir(saveDir):
            rmtree(saveDir)
        os.mkdir(saveDir)
        # Create a directory for every geometry
        os.mkdir(os.path.join(saveDir, 'points'))
        os.mkdir(os.path.join(saveDir, 'lines'))
        os.mkdir(os.path.join(saveDir, 'polygons'))
        # For every file in original directory see if the name includes a given geometry
        for name in os.listdir(datasetDir):
            src = os.path.join(datasetDir, name)
            tile_name = name.split('_')[0]
            metadata_file = 'Metadata' + '_' + tile_name + '.xml'
            metadata_file = os.path.join(datasetDir, metadata_file)
            if 'point' in name:
                dst = os.path.join(saveDir, 'points')
                copy2(src, dst)
                copy2(metadata_file, dst)
            elif 'line' in name:
                dst = os.path.join(saveDir, 'lines')
                copy2(src, dst)
                copy2(metadata_file, dst)
            elif 'polygon' in name:
                dst = os.path.join(saveDir, 'polygons')
                copy2(src, dst)
                copy2(metadata_file, dst)
            else:
                print("No a geometry")
        print("Done!")

    else:
        print("Directory of file {} does not exists".format(datasetDir))



if __name__ == "__main__":
    datasetDir = 'Contours_Bristol/terrain-5_3327781'
    splitGeometriesShapes(datasetDir)
