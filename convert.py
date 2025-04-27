import os
import subprocess
import sys
import wave


# Function to convert .flac to .wav using ffmpeg
def convert_flac_to_wav(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    flac_files = [f for f in os.listdir(input_dir) if f.endswith(".flac")]

    for flac_file in flac_files:
        input_path = os.path.join(input_dir, flac_file)
        output_file = os.path.splitext(flac_file)[0] + ".wav"
        output_path = os.path.join(output_dir, output_file)

        # Use ffmpeg to convert .flac to .wav
        command = ["ffmpeg", "-i", input_path, output_path]
        try:
            subprocess.run(command, check=True)
            print(f"Converted: {flac_file} -> {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {flac_file}: {e}")
            continue

        # Check integrity of the converted .wav file
        check_wav_integrity(output_path)


# Function to check the integrity of a .wav file by opening it with the wave module
def check_wav_integrity(wav_file):
    try:
        with wave.open(wav_file, "rb") as wav:
            # If the file opens without error, it's valid
            print(f"Integrity check passed for: {wav_file}")
    except wave.Error as e:
        print(f"Integrity check failed for: {wav_file}. Error: {e}")


# Main function
def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python convert_flac_to_wav.py <input_directory> <output_directory>"
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        sys.exit(1)

    convert_flac_to_wav(input_dir, output_dir)


if __name__ == "__main__":
    main()
