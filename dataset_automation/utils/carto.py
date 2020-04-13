import os 

def removeTextXML(mapfile):
# This function removes all text labels and shields in the stylesheet.
    text = 'TextSymbolizer'
    shields = 'ShieldSymbolizer'
    #text = 'Parameter'
    newPath = os.path.join( os.environ['carto'], "no_text_style.xml")
    newFile = open( newPath, "w")
    count = 0
    with open( mapfile , 'r') as f:
        for line in f:
            if text not in line and shields not in line:
                newFile.write(line)
            else:
                count+=1
        newFile.close()
    print("Done, {} lines with {} text were removed".format(count, text))
    print("New file in ", newPath)
        

if __name__ == "__main__":
    mapfile = os.path.join( os.environ['carto'], 'style.xml')
    removeTextXML(mapfile)