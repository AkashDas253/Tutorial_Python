# Matplotlib Cheatsheet

## 1. Importing Matplotlib
- `import matplotlib.pyplot as plt`  # Import pyplot

## 2. Creating a Basic Plot
- `plt.plot(x, y)`  # Basic line plot
- `plt.title('Title')`  # Set title
- `plt.xlabel('X-axis Label')`  # Set x-axis label
- `plt.ylabel('Y-axis Label')`  # Set y-axis label
- `plt.show()`  # Display plot

## 3. Creating Multiple Plots
- `plt.subplot(nrows, ncols, index)`  # Create subplot
- `plt.plot(x1, y1, label='Line 1')`  # First plot
- `plt.plot(x2, y2, label='Line 2')`  # Second plot
- `plt.legend()`  # Show legend

## 4. Customizing Plots
- `plt.grid()`  # Add grid
- `plt.xlim(xmin, xmax)`  # Set x-axis limits
- `plt.ylim(ymin, ymax)`  # Set y-axis limits
- `plt.xticks(rotation=45)`  # Rotate x-axis tick labels

## 5. Bar Plots
- `plt.bar(x, height)`  # Vertical bar plot
- `plt.barh(y, width)`  # Horizontal bar plot
- `plt.title('Bar Plot Title')`  # Set title

## 6. Histograms
- `plt.hist(data, bins=10)`  # Create histogram
- `plt.title('Histogram Title')`  # Set title

## 7. Scatter Plots
- `plt.scatter(x, y)`  # Create scatter plot
- `plt.colorbar()`  # Add color bar

## 8. Saving Plots
- `plt.savefig('filename.png')`  # Save plot as file
- `plt.savefig('filename.pdf')`  # Save as PDF

## 9. Customizing Markers and Lines
- `plt.plot(x, y, marker='o', linestyle='--', color='r')`  # Customize marker and line style

## 10. Using Styles
- `plt.style.use('ggplot')`  # Use predefined style
- `plt.style.use('seaborn-darkgrid')`  # Another style

## 11. Subplots
- `fig, axs = plt.subplots(nrows, ncols)`  # Create subplots
- `axs[0, 0].plot(x1, y1)`  # Plot in subplot
- `axs[1, 1].scatter(x2, y2)`  # Scatter in subplot

## 12. Adding Annotations
- `plt.annotate('Annotation', xy=(x, y), xytext=(x_offset, y_offset), arrowprops=dict(facecolor='black'))`  # Add annotation

## 13. Pie Charts
- `plt.pie(sizes, labels=labels, autopct='%1.1f%%')`  # Create pie chart
- `plt.axis('equal')`  # Equal aspect ratio for pie chart

## 14. 3D Plots
- `from mpl_toolkits.mplot3d import Axes3D`  # Import 3D toolkit
- `fig = plt.figure()`  # Create a new figure
- `ax = fig.add_subplot(111, projection='3d')`  # Create 3D axes
- `ax.scatter(x, y, z)`  # 3D scatter plot

## 15. Customizing Ticks
- `plt.xticks(ticks=[1, 2, 3], labels=['A', 'B', 'C'])`  # Customize tick labels
- `plt.yticks(rotation=45)`  # Rotate y-axis tick labels

## 16. Box Plots
- `plt.boxplot(data)`  # Create box plot
- `plt.title('Box Plot Title')`  # Set title

## 17. Violin Plots
- `plt.violinplot(data)`  # Create violin plot
- `plt.title('Violin Plot Title')`  # Set title

## 18. Heatmaps
- `plt.imshow(data, cmap='hot', interpolation='nearest')`  # Create heatmap
- `plt.colorbar()`  # Add color bar

## 19. Error Bars
- `plt.errorbar(x, y, yerr=error, fmt='o')`  # Create plot with error bars

## 20. Logarithmic Scale
- `plt.xscale('log')`  # Set x-axis to logarithmic scale
- `plt.yscale('log')`  # Set y-axis to logarithmic scale

## 21. Polar Plots
- `plt.polar(theta, r)`  # Create polar plot

## 22. Step Plots
- `plt.step(x, y)`  # Create step plot

## 23. Fill Between
- `plt.fill_between(x, y1, y2)`  # Fill area between two curves

## 24. Quiver Plots
- `plt.quiver(X, Y, U, V)`  # Create quiver plot for vector fields

## 25. Stream Plots
- `plt.streamplot(X, Y, U, V)`  # Create stream plot for vector fields

## 26. Twin Axes
- `fig, ax1 = plt.subplots()`  # Create a figure and a set of subplots
- `ax2 = ax1.twinx()`  # Create a twin Axes sharing the x-axis
- `ax1.plot(x, y1, 'g-')`  # Plot on ax1
- `ax2.plot(x, y2, 'b-')`  # Plot on ax2

## 27. Broken Axis
- `fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)`  # Create subplots with shared x-axis
- `ax1.plot(x, y1)`  # Plot on ax1
- `ax2.plot(x, y2)`  # Plot on ax2
- `ax1.set_ylim(y1_min, y1_max)`  # Set y-axis limits for ax1
- `ax2.set_ylim(y2_min, y2_max)`  # Set y-axis limits for ax2

## 28. Annotating Subplots
- `fig, ax = plt.subplots()`  # Create a figure and a set of subplots
- `ax.plot(x, y)`  # Plot data
- `ax.annotate('Annotation', xy=(x, y), xytext=(x_offset, y_offset), arrowprops=dict(facecolor='black'))`  # Add annotation

## 29. Adding Text
- `plt.text(x, y, 'Text', fontsize=12)`  # Add text at specified coordinates

## 30. Customizing Legends
- `plt.legend(loc='upper right', fontsize='small', title='Legend Title')`  # Customize legend location, font size, and title

## 31. Customizing Colors
- `plt.plot(x, y, color='#FF5733')`  # Use hex color codes
- `plt.plot(x, y, color='C0')`  # Use color cycle

## 32. Customizing Line Styles
- `plt.plot(x, y, linestyle='-', linewidth=2)`  # Customize line style and width

## 33. Customizing Markers
- `plt.plot(x, y, marker='o', markersize=10, markerfacecolor='blue', markeredgewidth=2, markeredgecolor='red')`  # Customize marker style

## 34. Customizing Fonts
- `plt.title('Title', fontsize=20, fontweight='bold', fontname='Times New Roman')`  # Customize font properties

## 35. Adding a Secondary Y-axis
- `fig, ax1 = plt.subplots()`  # Create a figure and a set of subplots
- `ax2 = ax1.twinx()`  # Create a twin Axes sharing the x-axis
- `ax1.plot(x, y1, 'g-')`  # Plot on ax1
- `ax2.plot(x, y2, 'b-')`  # Plot on ax2
- `ax1.set_ylabel('Y1')`  # Set y-axis label for ax1
- `ax2.set_ylabel('Y2')`  # Set y-axis label for ax2