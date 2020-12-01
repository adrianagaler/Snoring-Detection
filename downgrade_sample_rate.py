import os   
from pydub import AudioSegment

def main(directory, save_directory):
    extensions = ["/1", "/0"]
    for extension in extensions:
        for filename in os.listdir(directory + extension):
            
            if filename.endswith(".wav"): 
                sound = AudioSegment.from_wav(directory + extension + "/" + filename)
                sound = sound.set_frame_rate(16000)
                sound.export(save_directory + extension + "/" + filename, format="wav")                 
    

                

directory = "/Users/ariadnarotaru/Desktop/249r/Snoring_Dataset"
save_directory = "/Users/ariadnarotaru/Desktop/249r/Snoring_Dataset_@16000"
main(directory, save_directory)

