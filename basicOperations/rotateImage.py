import cv2
import argparse

def rotate_image(image_path, angle):
    # Load the image
    image = cv2.imread(image_path)

    if image is not None:
        # Get the image dimensions
        height, width = image.shape[:2]

        # Calculate the rotation matrix
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply the rotation transformation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        cv2.imshow('Rotated Image', rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to load the image.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rotate an image.')
    parser.add_argument('image_path', type=str, default='dogs.jpeg', nargs='?',help='Path to the image')
    parser.add_argument('--angle', type=float, default=45, help='Rotation angle in degrees (default: 45)')

    args = parser.parse_args()
    rotate_image(args.image_path, args.angle)

