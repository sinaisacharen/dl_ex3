import os

def low_case_name_file(midi_folder):
    """rename the name of the folder in order to import the name of the files.

    Args:
        midi_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    # List all files in the folder
    files = os.listdir(midi_folder)

    # Loop over all files in the folder
    for file in files:
        # Check if the file is a MIDI file
        if file.endswith('.mid'):
            # Get the current file path
            old_file_path = os.path.join(midi_folder, file)
            
            # Convert the file name to lowercase
            new_file_name = file.lower()
            new_file_path = os.path.join(midi_folder, new_file_name)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{file}' to '{new_file_name}'")

    return print("All MIDI files have been renamed to lowercase.")