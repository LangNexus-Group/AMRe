import json
import os

# Directory paths
mvsa_dir = "/root/autodl-tmp/AVDataModel/data/dataset/MVSA/MVSA_Single"
data_dir = os.path.join(mvsa_dir, "data")

# Input and output file mappings
file_mappings = {
    "my_train.txt": "train.jsonl",
    "my_val.txt": "val.jsonl",
    "my_test.txt": "test.jsonl"
}

# Process each file in file_mappings
for input_file, output_jsonl in file_mappings.items():
    input_path = os.path.join(mvsa_dir, input_file)
    output_path = os.path.join(mvsa_dir, output_jsonl)

    with open(input_path, "r") as f_labels, open(output_path, "w") as f_jsonl:
        for line in f_labels:
            # Each line contains "image_name label"
            line = line.strip()
            if not line:
                continue

            img_name, image_label = line.split(" ")
            ID = img_name.split(".")[0]  # Extract ID from image name

            # Read the caption from the corresponding .txt file
            text_file = os.path.join(data_dir, f"{ID}.txt")
            try:
                with open(text_file, "r", encoding="utf-8") as f_text:
                    caption = f_text.read().strip()
            except UnicodeDecodeError:
                with open(text_file, "r", encoding="ISO-8859-1") as f_text:
                    caption = f_text.read().strip()

            # Construct the JSON object]
            img_path = os.path.join("data",img_name)
            json_obj = {
                "img": img_path,           # Image filename
                "text": caption,           # Text content
                "image_label": image_label # Image sentiment label
            }

            # Write JSON object to .jsonl file
            f_jsonl.write(json.dumps(json_obj) + "\n")

    print(f"Created {output_path}")