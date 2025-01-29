import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os
import requests

def DMD(snapshots, rank=10):
    """
    Performs DMD on snapshot data, where time is the last axis
    and snapshots are separated by a constant time interval.

    Parameters
    ----------
    snapshots: Snapshot matrix. Can be multidimensional, but time first be the last axis.
    
    Returns
    -------
    eigval: eigenvalues corresponding to the DMD_modes.
    DMD_modes: Matrix of DMD_modes (space, mode_index).
    """
    orig_shape = snapshots.shape[:-1]
    if snapshots.ndim != 2:
        snapshots_flattened = snapshots.reshape((np.prod(orig_shape), -1))
    else:
        snapshots_flattened = snapshots
    X = snapshots_flattened[:,:-1]
    Y = snapshots_flattened[:,1:]
    U, S, VH = sp.linalg.svds(X, k=rank)
    Abar = U.conj().T@Y@VH.conj().T@np.diag(1/S)
    eigval, eigvec = sp.linalg.eigs(Abar, k=rank)
    DMD_modes = Y@VH.conj().T@np.diag(1/S)@eigvec
    DMD_modes = DMD_modes.T
    DMD_modes = DMD_modes.reshape((-1,) + orig_shape)
    return eigval, DMD_modes
    
def POD(X, weight=None):
    """
    Computes the POD using the method of snapshots.
    
    Parameters
    ----------
    X: Snapshot matrix. Can be multidimensional, but time first be the last axis.
    weight: Weight matrix for weighting the snapshots.
    
    Returns
    -------
    eigval: eigenvalues corresponding to the pod_modes.
    pod_modes: Matrix of pod_modes (space, mode_index).
    """
    # Store the spatial shape
    orig_shape = X.shape[:-1]
    if X.ndim != 2:
        # Must flatten spatial dimensions before SVD
        X_snaps = X.reshape((np.prod(orig_shape), -1))
    else:
        X_snaps = X
    # Form the covariance matrix
    C = X_snaps.T@X_snaps
    # Perform the eigenvalue decompositions
    eigval, eigvec = sp.linalg.eigs(C, k=24)
    # Reconstruct the POD modes
    pod_modes = X_snaps@eigvec
    # Make mode_index the first dimension
    pod_modes = pod_modes.T
    # Unflatten the spatial dimension
    pod_modes = pod_modes.reshape((-1,) + orig_shape)
    return eigval, pod_modes

def plot_ODE_data(ode_data):
    """
    Plots ode data (either 2D or 3D).

    Parameters
    ----------
    ode_data: np.array with shape (number of snapshots, number of dimensions).
    """
    dim = ode_data.shape[-1]
    # Plot the data
    if dim == 3:
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(ode_data[:, 0], ode_data[:, 1], ode_data[:, 2])
    else:
        ax = plt.subplot()
        ax.plot(ode_data[:, 0], ode_data[:, 1])
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    if dim == 3:
        ax.set_zlabel(r'z')

def plot_cylinder_data(x, y, cylinder_data, fig_ax=None, cmap=None):
    """
    Plots a snapshot of flow past a cylinder.

    Parameters
    ----------
    x: np.array with x-coordinates.
    y: np.array with y-coordinates.
    cylinder_data: np.array with shape (x resolution, y resolution).
    fig_ax: tuple (fig, ax) obtained with subplot.
    cmap: matplotlib colormap.
    """
    if fig_ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig, ax = fig_ax
    if cmap is None:
        cmap='plasma'
    circle = plt.Circle((0, 0), 0.5, color='grey')
    cb = ax.contourf(x, y, cylinder_data, levels=40, cmap=cmap)
    ax.set_xlim([-1, 8])
    ax.set_ylim([-2, 2])
    ax.add_patch(circle)
    ax.set_aspect(True)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    fig.colorbar(cb, ax=ax)

def download_data():
    """
    Download data for the tutorial from Hugging Face. Creates a directory
    called 'data' in the current working directory.
    """
    # Create directory 'data' if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created directory 'data'.")
    else:
        print("Directory 'data' already exists.")

    # List of file URLs
    file_urls = [
        "https://huggingface.co/datasets/CEMAC/cylinder_vortex_shedding/resolve/main/cylinder_flow_data.npz",
        "https://huggingface.co/datasets/CEMAC/cylinder_vortex_shedding/resolve/main/cylinder_mass_u.npz",
        "https://huggingface.co/datasets/CEMAC/cylinder_vortex_shedding/resolve/main/cylinder_mass_v.npz",
        "https://huggingface.co/datasets/CEMAC/cylinder_vortex_shedding/resolve/main/equation_A.npz",
        "https://huggingface.co/datasets/CEMAC/cylinder_vortex_shedding/resolve/main/equation_B.npz",
        "https://huggingface.co/datasets/CEMAC/cylinder_vortex_shedding/resolve/main/equation_C.npz",
        "https://huggingface.co/datasets/CEMAC/cylinder_vortex_shedding/resolve/main/lorentz.npz",
    ]

    # Download each file and save it to the 'data' directory if it doesn't already exist
    for url in file_urls:
        file_name = os.path.join('data', os.path.basename(url))
        if not os.path.exists(file_name):
            response = requests.get(url)
            with open(file_name, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {file_name}")
        else:
            print(f"{file_name} already exists.")

    print("All files have been checked and downloaded if necessary.")