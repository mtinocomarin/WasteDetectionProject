from ultralytics import YOLO

# Load your YOLO model
model = YOLO(r'C:\Users\milto\OneDrive\Desktop\413FinalProject\runs\detect\train\weights\best.pt')

# Run inference on an image
results = model.predict(source=r'C:\Users\milto\OneDrive\Desktop\Testing Trash\oldbana.jpg', save=True)

# Process each result (optional)
for result in results:
    print(result.boxes)  # Display the bounding boxes
    print(result.names)  # Display the class names

# Save the results if needed
print("Results saved to:", results[0].save_dir)
