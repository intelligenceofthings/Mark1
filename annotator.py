import requests
import json
import cv2
import matplotlib.pyplot as plt
import uuid
from metadata import metadata

from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env
OWLVIT_API_KEY = os.getenv("OWLVIT_API_KEY")

def annotate(objects, image_path):
    url = "https://api.va.landing.ai/v1/tools/text-to-object-detection"

    # Open image file
    with open(image_path, "rb") as image_file:
        files = {"image": image_file}
        
        # Define request data
        data = {
            "prompts": objects,
            "model": "owlv2"
        }

        # Set headers with proper authorization key
        headers = {
            "Authorization": "Basic "+OWLVIT_API_KEY
        }

        # Send the request
        response = requests.post(url, files=files, data=data, headers=headers)
    
    response_data = response.json()

    # Save the response data in metadata.py
    with open("metadata.py", "w") as f:
        f.write(f"metadata = {json.dumps(response_data, indent=4)}\n")

    print("Response saved in metadata.py")

def parser(label_counts):
    """Filters metadata to retain only the top-N scoring objects per label."""
    
    # Ensure metadata follows the correct structure
    if not isinstance(metadata["data"], list) or not metadata["data"]:
        print("Error: Metadata structure is incorrect.")
        return {"data": []}

    # Flatten metadata["data"] if it's nested
    all_items = metadata["data"][0] if isinstance(metadata["data"][0], list) else metadata["data"]

    # Ensure each object has a UUID (but do not modify existing ones)
    for item in all_items:
        if "id" not in item:
            item["id"] = str(uuid.uuid4())  # Assign new ID only if missing

    def get_top_uuids_by_score(all_items, label_counts):
        top_uuids = {}
        label_items = {}

        # Group items by label
        for item in all_items:
            label = item["label"]
            if label not in label_items:
                label_items[label] = []
            label_items[label].append(item)

        # Select only the top-N highest-scoring items per label
        for label, count in label_counts.items():
            if label in label_items:
                sorted_items = sorted(label_items[label], key=lambda x: x["score"], reverse=True)
                top_uuids[label] = [item["id"] for item in sorted_items[:count]]  # Take only the top-N objects

        return top_uuids

    # Get the top N UUIDs for each label
    top_uuids = get_top_uuids_by_score(all_items, label_counts)

    # Create new metadata containing only the selected top-N objects
    new_metadata = {"data": [[]]}  # Ensure it follows the correct structure

    for label, uuids in top_uuids.items():
        filtered_items = [
            {key: value for key, value in item.items() if key != "id"}  # Remove 'id' from output
            for item in all_items
            if item["label"] == label and item["id"] in uuids  # Ensure only top-N objects are included
        ]
        new_metadata["data"][0].extend(filtered_items)  # Append correctly

    return new_metadata


# Global variables for dragging
dragging = False
selected_box = None
offset_x = 0
offset_y = 0

def visualize(image_path, new_metadata):
    global dragging, selected_box, offset_x, offset_y

    plotted_objects = []  # Store plotted object names

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Instructions text
    instructions = "Press 'r' to Refresh, 'm' for Manual Annotation, 'a' to Accept"

    all_detections = new_metadata["data"]
    colors = {"fan": (0, 255, 0), "light": (0, 255, 0)}

    bounding_boxes = []  # Store bounding box information
    label_dict = {}

    for detections in all_detections:
        for obj in detections:
            label = obj["label"]
            if label not in label_dict:
                label_dict[label] = []
            label_dict[label].append(obj)

    # Assign sequential naming (fan1, fan2, light1, light2)
    for label, detections in label_dict.items():
        detections.sort(key=lambda obj: (obj["bounding_box"][0], obj["bounding_box"][1]))
        for idx, obj in enumerate(detections):
            x1, y1, x2, y2 = map(int, obj["bounding_box"])
            label_with_suffix = f"{label}{idx + 1}"
            bounding_boxes.append({"label": label_with_suffix, "box": [x1, y1, x2, y2], "color": colors.get(label.lower(), (0, 255, 255))})
            plotted_objects.append(label_with_suffix)  # Store plotted label

    # Mouse callback function for dragging
    def on_mouse(event, x, y, flags, param):
        global dragging, selected_box, offset_x, offset_y

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if mouse clicked inside any bounding box
            for box in bounding_boxes:
                x1, y1, x2, y2 = box["box"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_box = box
                    dragging = True
                    offset_x = x - x1
                    offset_y = y - y1
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging and selected_box:
                # Update the bounding box coordinates while dragging
                x1, y1, x2, y2 = selected_box["box"]
                width = x2 - x1
                height = y2 - y1
                selected_box["box"] = [x - offset_x, y - offset_y, x - offset_x + width, y - offset_y + height]

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            selected_box = None  # Deselect box after releasing mouse

    cv2.namedWindow("Annotated Image")
    cv2.setMouseCallback("Annotated Image", on_mouse)

    while True:
        temp_image = image.copy()

        # Draw bounding boxes on the image
        for box in bounding_boxes:
            x1, y1, x2, y2 = box["box"]
            label = box["label"]
            color = box["color"]

            # Draw bounding box
            cv2.rectangle(temp_image, (x1, y1), (x2, y2), color, 2)

            # Position the label text
            text_y = max(y1 - 5, 10)
            cv2.putText(temp_image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black outline
            cv2.putText(temp_image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Colored text

        # Display instructions only on the screen, not in the saved image
        display_image = temp_image.copy()
        cv2.putText(display_image, instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow("Annotated Image", cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            print("Refreshing annotations...")
            cv2.destroyAllWindows()
            return "refresh", plotted_objects
        elif key == ord('m'):
            print("Entering manual annotation mode...")
            return "manual", plotted_objects
        elif key == ord('a'):
            print("Accepting and saving image.")
            cv2.imwrite("annotated_image.jpg", cv2.cvtColor(temp_image, cv2.COLOR_RGB2BGR))  # Save without instructions
            cv2.destroyAllWindows()
            return "accept", plotted_objects

    
def user_choice(image_path, objects, label_counts):
    while True:
        new_metadata = parser(label_counts)
        action, plotted_objects = visualize(image_path, new_metadata)  # Capture plotted objects

        #print("🖼️ Final Plotted Objects:", plotted_objects)  # Debug print

        if action == "refresh":
            annotate(objects, image_path)
        elif action == "manual":
            print("Drag and drop functionality needs a GUI-based editor.")
        elif action == "accept":
            return plotted_objects  # Return the plotted object list


def start_annot(objects, image_path, label_counts):
    annotate(objects, image_path)
    return user_choice(image_path, objects, label_counts)