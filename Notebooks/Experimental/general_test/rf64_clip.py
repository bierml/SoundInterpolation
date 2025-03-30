import struct
import random
import numpy as np
# --- User parameters ---
source_file_path = "podcastslong.rf64"         # Path to the source RF64 file
destination_file_path = "podcastslongс.rf64"    # Output file path
clip_thrshd_db_min = 5          # e.g., clip at -10 dB (negative value)
clip_thrshd_db_max = 15
chunk_size = 44032                    # Number of bytes to process per chunk

def db_to_multiplicator(dbvalue):
    return 10 ** (-dbvalue / 20.0)

# Compute amplitude multiplier:
# For PCM, full-scale for 16-bit is 32768.
# A threshold of -10 dB means amplitude factor = 10^(-10/20)

def read_chunk_header(f):
    hdr = f.read(8)
    if len(hdr) < 8:
        return None, None
    tag, size = struct.unpack('<4sL', hdr)
    return tag, size

# --- Open the source file and parse header ---
with open(source_file_path, 'rb') as fin:
    # Read first 12 bytes: signature, file size, and "WAVE"
    riff_header = fin.read(12)
    if len(riff_header) < 12:
        raise ValueError("File too short")
    signature = riff_header[:4]
    if signature != b'RF64':
        raise ValueError("Not an RF64 file (signature = {})".format(signature))
    # We'll collect header bytes in this variable.
    header_bytes = riff_header
    data_chunk_found = False
    data_chunk_size = None

    # Read chunks until we hit the "data" chunk.
    while not data_chunk_found:
        tag, size = read_chunk_header(fin)
        if tag is None:
            raise ValueError("Reached end of file without finding data chunk")
        header_bytes += struct.pack('<4sL', tag, size)
        if tag == b'ds64':
            ds64_data = fin.read(size)
            header_bytes += ds64_data
        elif tag == b'fmt ':
            fmt_data = fin.read(size)
            header_bytes += fmt_data
            # For this example, we assume 16-bit PCM, mono.
            sampwidth = 2  # bytes per sample
            n_channels = 1
        elif tag == b'data':
            data_chunk_found = True
            data_chunk_size = size
            # Do not read any data now; we stop here.
        else:
            # For any other chunk, copy it entirely.
            chunk_data = fin.read(size)
            header_bytes += chunk_data

    if data_chunk_size is None:
        raise ValueError("Data chunk not found in source file")

    print("Header length:", len(header_bytes))
    print("Data chunk size (bytes):", data_chunk_size)

    # At this point, fin is positioned at the beginning of the data chunk.
    # --- Open destination file for writing ---
    with open(destination_file_path, 'wb') as fout:
        # Write the header exactly as read.
        fout.write(header_bytes)
        # Process the data chunk in pieces.
        bytes_remaining = data_chunk_size

        # For 16-bit PCM, each sample is 2 bytes.
        full_scale = 32768
        #print("Clip threshold (int):", clip_threshold)

        while bytes_remaining > 0:
            dbval = random.uniform(clip_thrshd_db_min,clip_thrshd_db_max)
            clip_threshold = int(db_to_multiplicator(dbval) * full_scale)
            to_read = min(chunk_size, bytes_remaining)
            raw_chunk = fin.read(to_read)
            if not raw_chunk:
                break
            # Convert raw bytes to a NumPy array.
            # This example assumes 16-bit PCM.
            samples = np.frombuffer(raw_chunk, dtype=np.int16)
            # Clip the sample values.
            clipped = np.clip(samples, -clip_threshold, clip_threshold)
            # Write the processed chunk (as bytes) to the output.
            fout.write(clipped.tobytes())
            bytes_remaining -= len(raw_chunk)
        # Do not copy trailing data: the header already contains all non-data chunks.

print("Processing complete. Processed file saved as:", destination_file_path)
