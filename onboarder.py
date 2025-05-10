import nl_processor
import annotator
import pyttsx3
import transcriptor
import spatial_inferencer

def text_to_speech(text):
    print(text)
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def onboarding_system(file_path):
    system = "Welcome to the onboarding system of InOT. Please let me know the smart devices in the home."
    text_to_speech(system)
    user = transcriptor.transcribe()
    image_path = file_path
    obj_str = nl_processor.refine(user)
    text_to_speech(f'I have recieved {obj_str}')
    obj_dic = eval(obj_str)

    obj_list = list(obj_dic.keys())

    print(obj_dic)

    text_to_speech("Starting the Fully Automatic Annotation Process...")

    object_list = annotator.start_annot(obj_list,image_path,obj_dic)
    text_to_speech("Annotation Complete! Moving to Spatial Reasoning System.")
    print(object_list)

    spatial = spatial_inferencer.information(object_list)

    with open("spatial_information.txt", "w") as file:
        file.write(spatial)

    print("Spatial Data Saved.")
    return spatial