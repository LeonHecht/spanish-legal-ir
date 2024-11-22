import numpy as np
import os

def save_attention_mask_to_file(attention_mask, file_path="attention_mask.txt"):
    """
    Save the attention mask to a text file for inspection.

    :param attention_mask: The attention mask tensor (PyTorch).
    :param file_path: File path to save the matrix.
    """
    # Convert the tensor to a NumPy array for easier handling
    attention_mask_np = attention_mask.cpu().numpy()
    
    # check if filepath already exists
    if os.path.exists(file_path):
        return

    # Save the array to a text file
    with open(file_path, "w") as f:
        for i, matrix in enumerate(attention_mask_np):
            f.write(f"Attention Mask for Sequence {i}:\n")
            np.savetxt(f, matrix, fmt="%d")
            f.write("\n\n")  # Add spacing between different matrices

    print(f"Attention mask saved to {file_path}")
