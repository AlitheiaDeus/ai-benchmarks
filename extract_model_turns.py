import json
import os
import glob

def aggregate_model_turns(input_folder, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as out_file:
        file_pattern = os.path.join(input_folder, "*.json")
        
        for file_path in glob.glob(file_pattern):
            if file_path == output_filepath:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    chat_data = json.load(file)
            except (json.JSONDecodeError, IsADirectoryError):
                continue
                
            model_aggregations = {}
            messages = chat_data.get("messages", [])
            
            for msg in messages:
                selected_index = msg.get("currentlySelected", 0)
                versions = msg.get("versions", [])
                
                if not versions or selected_index >= len(versions):
                    continue
                    
                active_version = versions[selected_index]
                
                if active_version.get("role") == "assistant":
                    sender_info = active_version.get("senderInfo", {})
                    model_name = sender_info.get("senderName", "Unknown_Model")
                    
                    extracted_text = []
                    
                    if "steps" in active_version:
                        for step in active_version["steps"]:
                            if step.get("type") == "contentBlock":
                                if step.get("style", {}).get("type") == "thinking":
                                    continue
                                for content_item in step.get("content", []):
                                    if content_item.get("type") == "text":
                                        extracted_text.append(content_item.get("text", ""))
                                        
                    elif "content" in active_version:
                        for content_item in active_version["content"]:
                            if content_item.get("type") == "text":
                                extracted_text.append(content_item.get("text", ""))
                                
                    combined_turn = "\n".join(extracted_text).strip()
                    
                    if combined_turn:
                        if model_name not in model_aggregations:
                            model_aggregations[model_name] = []
                        model_aggregations[model_name].append(combined_turn)

            if model_aggregations:
                filename = os.path.basename(file_path)
                for model, turns in model_aggregations.items():
                    out_file.write(f"========== File: {filename} | Model: {model} ==========\n\n")
                    for turn in turns:
                        out_file.write(f"{turn}\n\n")

if __name__ == "__main__":
    target_directory = "." 
    output_destination = "aggregated_model_turns.txt"
    aggregate_model_turns(target_directory, output_destination)