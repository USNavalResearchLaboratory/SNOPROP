# The source code has been authored by federal and non-federal employees. To the extent that a federal employee is an author of a portion of this software or a derivative work thereof, no copyright is claimed by the United States Government, as represented by the Secretary of the Navy ("GOVERNMENT") under Title 17, U.S. Code. All Other Rights Reserved. To the extent that a non-federal employee is an author of a portion of this software or a derivative work thereof, the work was funded in whole or in part by the U.S. Government, and is, therefore, subject to the following license: The Government has unlimited rights to use, modify, reproduce, release, perform, display, or disclose the computer software and computer software documentation in whole or in part, in any manner and for any purpose whatsoever, and to have or authorize others to do so. Any other rights are reserved by the copyright owner.

# Neither the name of NRL or its contributors, nor any entity of the United States Government may be used to endorse or promote products derived from this software, nor does the inclusion of the NRL written and developed software directly or indirectly suggest NRL's or the United States Government's endorsement of this product. 

# THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

import os
import csv
import pickle
import shutil

class FileIO:
    dataFolderName = "data/" 
    logFolderName = "logs/"
    dataDir = ""
    folder = ""

    def __init__(self):
        #folder = ""
        #folderFinal = folderTry
        #i = 1
        #while os.path.isdir(folderFinal):
        #    folderFinal = folderTry+"_"+str(i)
        #    i+=1
        #print("Saving simulation in folder:",folderFinal)
        #os.makedirs(folderFinal)
        #self.folder = folderFinal+"/"
        self.folder = ""# Just make the simulation folders in the local directory
        self.dataDir = self.folder + self.dataFolderName
        self.logDir = self.folder + self.logFolderName

        self.simFolders = [self.dataDir, self.logDir]
        self.simFolders = list(set(self.simFolders)) # Select unique (in case there are duplicates)
        if any([os.path.exists(f) for f in self.simFolders]): # If any folders already exist
            askDelete = input("Simulation folders already exist. Delete them? (y/n)")
            #askDelete = "y"
            print("input is "+askDelete)
            if askDelete == "y":
                print("Deleting existing folders")
                for f in self.simFolders:
                    if os.path.exists(f):
                        shutil.rmtree(f)
            else:
                print("Cannot run simulation because folders already exist. Quitting.")
                quit() 
        for f in self.simFolders:
            os.makedirs(f)

    def saveTextFile(self, textArr, fname):
        fname = self.logDir+fname+".txt"
        if os.path.exists(fname):
            print("Could not save file",fname)
        else:
            f = open(fname,"w")
            for row in textArr:
                f.write(row)
                f.write('\n')
            f.close()

    def saveDataTextFile(self, textArr, fname):
        fname = self.dataDir+fname+".txt"
        if os.path.exists(fname):
            print("Could not save file",fname)
        else:
            f = open(fname,"w")
            for row in textArr:
                f.write(row)
                f.write('\n')
            f.close()

    def makeCSVFile(self, fname, labels):
        newfname = self.dataDir+fname+'.csv'
        with open(newfname,'w') as f:
            f.write(','.join([str(e) for e in labels]))
        return newfname

    def appendRow(self, fname, numbers):
        with open(fname,'a') as f:
            f.write('\n'+','.join([str(e) for e in numbers]))

    def saveDataFile(self, frameData, fname, labels=None, info=None):
        fname = self.dataDir+fname+".csv"
        if os.path.exists(fname):
            print("Could not save file",fname)
        else:
            if labels is None:
                f = open(fname,"w", newline='')
                writer = csv.writer(f)
                for row in frameData:
                    writer.writerow(row)
                f.close()
            else:
                f = open(fname,"w", newline='')
                writer = csv.writer(f)
                writer.writerow([info])
                writer.writerow(labels)
                for row in frameData:
                    writer.writerow(row)
                f.close()
    def savePickle(self, data, fname):
        with open(self.dataDir+fname+'.pckl', 'wb') as f:
            pickle.dump(data, f)
