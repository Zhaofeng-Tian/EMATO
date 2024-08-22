import numpy as np  
from matplotlib.patches import Polygon

def plot_cars(ax, cars, if_arrow=False, if_plot_future = False):
    for car in cars:
        # Plot the current position of the car
        rotated_corners = car.get_rotated_corners()
        rect = Polygon(rotated_corners, edgecolor='black', facecolor='lightblue')
        
        x, y, yaw = car.pose_history[-1]

        # Plot an arrow to indicate the car's heading direction if needed
        if if_arrow:
            ax.arrow(
                x, y,
                car.speed * np.cos(yaw),
                car.speed * np.sin(yaw),
                head_width=0.5, head_length=0.6, fc='red', ec='red', alpha=0.7
            )

        # Plot the predicted future trajectory
        alpha_list = np.linspace(0.1, 0.6, len(car.future_traj))
        i = 0
        # for future_point in car.future_traj:
        if if_plot_future:
            for future_point in car.future_traj[::-1]:
            
                future_x, future_y, future_yaw = future_point

                # Define the unrotated rectangle corners relative to the car's center for the future position
                half_length = car.length / 2.0
                half_width = car.width / 2.0
                corners = np.array([
                    [half_length, half_width],  # front-left
                    [half_length, -half_width],  # front-right
                    [-half_length, -half_width],  # rear-right
                    [-half_length, half_width]   # rear-left
                ])

                # Rotation matrix based on future yaw angle
                rotation_matrix = np.array([
                    [np.cos(future_yaw), -np.sin(future_yaw)],
                    [np.sin(future_yaw), np.cos(future_yaw)]
                ])

                # Rotate and translate corners to the global position for the future point
                rotated_corners = corners @ rotation_matrix.T + [future_x, future_y]

                # Plot the polygon for the future position
                future_rect = Polygon(rotated_corners, edgecolor='grey', facecolor='lightblue', alpha=alpha_list[i])
                ax.add_patch(future_rect)
                i += 1
            ax.add_patch(rect)
        else:
            ax.add_patch(rect)