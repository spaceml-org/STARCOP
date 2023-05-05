import numpy as np
from pylab import plt
from .normalizer_module import DataNormalizer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os 

class Visualizer:

    def __init__(self):
        self.visualization_three_bands = [0, 1, 2] # use the first three bands for viz

        """
        - data visualization
        class Visualizer
            has the same Normalizer
            can convert data x to RGB (alike)
            can convert data y to visualization
        ^ ModelHandler has the Visualizer and uses it when logging 
        ^ it has it from the outside though, because I also want to use it 
                        before to provide some data inspections / debug
        """

        pass

    def x_to_image(self, x):
        # select three bands, normalize them to easy visual range
        x = np.clip(x / np.max(x),0,1)

        number_of_channels = x.shape[0]
        if number_of_channels < len(self.visualization_three_bands):
            x = x[[0],:,:] # fall back to visualizing just one band ...
        else:
            x = x[self.visualization_three_bands,:,:]

        return x

    def y_to_image(self, y):
        # keep normalized instead of 0-5000 I'd rather see 0-1
        return y
    
    def plot_x_y_pred(self, x, y, pred, show_colorbar = True):
        x = self.x_to_image(x)
        y = self.y_to_image(y)
        pred = self.y_to_image(pred)
            
        figure = plt.figure(figsize=(8, 4))

        img = np.moveaxis(x, 0, -1)
        label = np.moveaxis(y, 0, -1)
        prediction = np.moveaxis(pred, 0, -1)

        figure.add_subplot(1, 3, 1)
        plt.axis("off")
        plt.imshow(img)

        figure.add_subplot(1, 3, 2)
        plt.axis("off")
        im = plt.imshow(label[:,:,0]) # , vmin=0, vmax=1)

        if show_colorbar:
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, format='%2.2f')


        figure.add_subplot(1, 3, 3)
        plt.axis("off")
        im = plt.imshow(prediction[:,:,0]) # , vmin=0, vmax=1)

        if show_colorbar:
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, format='%2.2f')

        return plt        

    def debug_data_model(self, data_module, settings_debug):
        # works as a demo ...
        train_dataloader = data_module.train_dataloader()

        os.makedirs("debugs", exist_ok=True)
        for batch in train_dataloader:
            xs, ys = batch["input"],batch["output"]
            for idx in range(min(len(xs), settings_debug.debug_visualized_save_how_many_xy)):
                x = xs[idx].numpy()
                y = ys[idx].numpy()

                x = self.x_to_image(x)
                y = self.y_to_image(y)

                figure = plt.figure(figsize=(8, 4))

                img = np.moveaxis(x, 0, -1)
                label = np.moveaxis(y, 0, -1)

                figure.add_subplot(1, 2, 1)
                plt.axis("off")
                plt.imshow(img)

                figure.add_subplot(1, 2, 2)
                plt.axis("off")
                plt.imshow(label[:,:,0]) # , vmin=0, vmax=1)
                plt.savefig("debugs/demo_"+str(idx).zfill(3)+".png")
                # plt.show()
                plt.close()
        
        