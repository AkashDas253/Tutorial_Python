# Pillow Cheatsheet

## 1. Importing Pillow
- from PIL import Image  # Import Image module
- from PIL import ImageDraw  # Import ImageDraw for drawing
- from PIL import ImageFont  # Import ImageFont for text

## 2. Opening and Displaying Images
- img = Image.open('image.jpg')  # Open an image file
- img.show()  # Display image

## 3. Saving Images
- img.save('new_image.png')  # Save image in a different format
- img.save('image.jpg', quality=85)  # Save with specific quality

## 4. Creating a New Image
- img = Image.new('RGB', (width, height), color='white')  # Create a new image

## 5. Image Size and Format
- img.size  # Get image size (width, height)
- img.format  # Get image format

## 6. Resizing Images
- img_resized = img.resize((new_width, new_height))  # Resize image
- img_resized = img.thumbnail((max_width, max_height))  # Resize with thumbnail

## 7. Cropping Images
- img_cropped = img.crop((left, upper, right, lower))  # Crop image

## 8. Rotating and Flipping Images
- img_rotated = img.rotate(angle)  # Rotate image
- img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip image horizontally

## 9. Converting Images
- img_gray = img.convert('L')  # Convert image to grayscale
- img_rgb = img.convert('RGB')  # Convert to RGB

## 10. Drawing on Images
- draw = ImageDraw.Draw(img)  # Create a drawing context
- draw.rectangle([(x1, y1), (x2, y2)], outline='black', fill='blue')  # Draw rectangle
- draw.text((x, y), 'Text', fill='white')  # Draw text

## 11. Adding Filters
- from PIL import ImageFilter  # Import ImageFilter
- img_blurred = img.filter(ImageFilter.BLUR)  # Apply blur filter
- img_sharpened = img.filter(ImageFilter.SHARPEN)  # Apply sharpen filter

## 12. Working with Transparency
- img_with_alpha = img.convert('RGBA')  # Convert to RGBA
- img_with_alpha.putalpha(128)  # Set transparency

## 13. Creating GIFs
- img.save('animation.gif', save_all=True, append_images=[img1, img2], duration=500, loop=0)  # Create GIF

## 14. Image Metadata
- exif_data = img._getexif()  # Get EXIF data

## 15. Image Enhancements
- from PIL import ImageEnhance  # Import ImageEnhance
- enhancer = ImageEnhance.Contrast(img)  # Create contrast enhancer
- img_enhanced = enhancer.enhance(factor)  # Enhance image contrast
