from Codebase.DataPlotter import DataPlotter
from Codebase.PINNCubeDataSet import PINNCubeDataSet
from Codebase.PINNCubeLossHandler import PINNCubeLossHandler

import matplotlib.pyplot as plt
import numpy as np

class CubeDataPlotter(DataPlotter):

    def plot_faces(self, dataset: PINNCubeDataSet, loss_handler: PINNCubeLossHandler, draw: bool=False, block: bool=False, filename: str=None):

        plot_info = {
            'nxminus': {
                'axis1': 1,
                'axis2': 2,
                'title': 'Nx-'
            },
            'nxplus': {
                'axis1': 1,
                'axis2': 2,
                'title': 'Nx+'
            },
            'nyminus': {
                'axis1': 2,
                'axis2': 0,
                'title': 'Ny-'
            },
            'nyplus': {
                'axis1': 2,
                'axis2': 0,
                'title': 'Ny+'
            },
            'nzminus': {
                'axis1': 0,
                'axis2': 1,
                'title': 'Nz-'
            },
            'nzplus': {
                'axis1': 0,
                'axis2': 1,
                'title': 'Nz+'
            }
        }

        rows, cols = 2, 3

        fig, axs = plt.subplots(rows, cols, figsize=(12, 7))

        losses = loss_handler.train_losses['main']
        bc_names = dataset.get_labels('bc', 'train')

        scatters = []
        min_loss = 0.0
        max_loss = -1.0

        for i, name in enumerate(bc_names):
            row = int(i % rows)
            col = int((i - row) / rows)

            loss = next((l for l in losses if l.name == name), None)
            if loss is not None:
                x, num, _ = dataset.get_data(name, 'bc', 'train')
                axis1 = plot_info[name]['axis1']
                axis2 = plot_info[name]['axis2']
                title = plot_info[name]['title']

                loss_values = np.linalg.norm(loss._eval_roots(x), axis=1)
                sct = axs[row, col].scatter(np.reshape(x[:, axis1], num), np.reshape(x[:, axis2], num),
                                      c=loss_values, cmap='RdBu_r', s=5)
                scatters.append(sct)
                max_loss = max(max_loss, np.max(loss_values))

                axs[row, col].set_title(title)
                axs[row, col].set_aspect('equal')

        for sct in scatters:
            sct.set_clim(min_loss, max_loss)

        fig.colorbar(scatters[-1], ax=axs.ravel().tolist())

        if filename is not None:
            plt.savefig(filename, dpi=200)
        if draw:
            plt.draw()
            plt.pause(1e-16)
        if block:
            plt.show(block=True)
        print('plot solution completed')

    def plot_data(self, dataset: PINNCubeDataSet,
                  draw=True, block=False, filename=None):

        x_data, num, _ = dataset.get_data('x_displacement', 'data', 'train')
        d_data, _, _ = dataset.get_data('displacement', 'data', 'train')
        x_displaced = x_data + d_data

        x_mesh = dataset.data_handler.get_mesh()
        d_mesh = dataset.data_handler.get_displacement()[0]
        x_mesh_displaced = x_mesh + d_mesh

        d_data_norm = np.linalg.norm(d_data, axis=1)
        d_mesh_norm = np.linalg.norm(d_mesh, axis=1)

        fig = plt.figure(figsize=(16, 5))

        # Add x, y gridlines
        def plot_view(ax, angle=None):
            ax.grid(b=True, color='grey',
                    linestyle='-.', linewidth=0.3,
                    alpha=0.2)

            ax.scatter3D(x_mesh_displaced[:,0], x_mesh_displaced[:,1], x_mesh_displaced[:,2], c='black', s=2, alpha=0.01)
            sct1 = ax.scatter3D(x_displaced[:,0], x_displaced[:,1], x_displaced[:,2], c=d_data_norm, cmap='RdBu_r', s=5)

            sct1.set_clim(np.min(d_mesh_norm), np.max(d_mesh_norm))

            if angle is None:
                ax.set_zlim(np.min(x_mesh_displaced[:,2]), np.max(x_mesh_displaced[:,2]))

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            if angle is not None:
                ax.view_init(elev=angle[0], azim=angle[1], roll=angle[2])

            return sct1

        ax = fig.add_subplot(1, 3, 1, projection='3d')
        sct1 = plot_view(ax)

        # YZ
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.set_proj_type('ortho')
        sct1 = plot_view(ax, [0, 0, 0])
        ax.set_xticklabels([])
        ax.set_xlabel('')

        # XY
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.set_proj_type('ortho')
        sct1 = plot_view(ax, [90, -90, 0])
        ax.set_zticklabels([])
        ax.set_zlabel('')

        fig.colorbar(sct1, ax=ax, shrink=0.6, aspect=8, pad=0.1)
        fig.suptitle('Data Points', fontsize=self.FONT_SIZE)

        self._finish(fig, title = None, filename = filename, draw = draw, block = block, dpi = 600)
        print('plot data points completed')