#import the basic libraries
import os
import librosa
import math
import json

#path of the dataset 
dataset_path = r'...\GitHub\Group5_DeepLearning\genres'
#empty .json file to store the data information
json_path = 'data_oficial.json'

#music parameters to generate the MFCCs 
SAMPLE_RATE = 22050 #frequency
DURATION = 30 #measured in seconds 
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

#save the tracks in differents segments (styles) 
def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

#dataset_path - path to take the dataset
#json_path - where to store the mfcc

    #dictornary to store data 
    data = {
        "mapping": [],
        "mfcc": [], #training inputs (tracks)
        "labels": []  #target (outputs) (classification of the tracks)
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vector_per_segment = math.ceil(num_samples_per_segment/hop_length) #ceil to make 1.2 -> 2 

    #loop through the genres 
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)): #unpack
        
        #ensure that we're not at the root level
        if dirpath is not dataset_path:
            
            #save the semantic label (the name of the classification)
            semantic_label = dirpath.split('/')[-1] #gender/blues => ['gender','blues']
            data['mapping'].append(semantic_label) #save the genrders in mapping dic
            print('\nProcessing: {}'.format(semantic_label))

            #go throught all the files in the fold
            for f in filenames: 

                #load the audio file
                file_path = os.path.join(dirpath, f) 
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                #process segments extracting mfcc and storing data 
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s # s=0 -> 0
                    finish_sample = start_sample + num_samples_per_segment  #s=0 -> num_samples_per_segment
                    
                    #analyse a slice of the signal
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr,
                                                n_mfcc = n_mfcc,
                                                n_fft = n_fft,
                                                hop_length = hop_length)
                    mfcc = mfcc.T #better to work with the transposed matrix 

                    #store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vector_per_segment:
                        data['mfcc'].append(mfcc.tolist()) 
                        data['labels'].append(i-1) #track name 
                        print('{}, segment:{}'.format(file_path, s+1))

    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)

if __name__ == '__main__':
    save_mfcc(dataset_path, json_path, num_segments=5)
