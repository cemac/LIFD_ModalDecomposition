import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import requests

def DMD(X, rank=10):
    """
    Performs DMD on snapshot data, where time is the last axis
    and snapshots are separated by a constant time interval.

    Parameters
    ----------
    snapshots: List of snapshot matrices. Each element is a snapshot matrix for a physical variable.
       Each matrix can be multidimensional, but time must be the last axis.
    rank: rank of the approximation of B
    
    Returns
    -------
    eigval: eigenvalues corresponding to the DMD_modes.
    DMD_modes: List of arrays of dmd modes. Each element of the list gives the dmd modes 
       for the corresponding physical variable given in X. Each dmd mode array has layout 
       (mode_index, variable, space).
    dmd_mode_amplitudes: Array of amplitudes of the dmd modes.
    """
    orig_shapes = []
    var_lengths = []
    A = []
    for var in X:
        orig_shapes.append(var.shape[:-1])
        var_lengths.append(np.prod(orig_shapes[-1]))
        if var.ndim != 2:
            A.append(var.reshape((var_lengths[-1], -1)))
        else:
            A.append(var)
    A = np.vstack(A)
    X = A[:, :-1]
    Y = A[:, 1:]
    U, S, VH = sp.linalg.svds(X, k=rank)
    Abar = U.conj().T @ Y @ VH.conj().T @ np.diag(1/S)
    eigval, eigvec = np.linalg.eig(Abar)
    dmd_modes = Y @ VH.conj().T @ np.diag(1/S) @ eigvec
    dmd_modes = dmd_modes.T
    indices = np.argsort(np.abs(eigval.imag))
    eigval = eigval[indices]
    dmd_modes = dmd_modes[indices]
    dmd_mode_amplitudes = np.linalg.lstsq(dmd_modes.T, A)[0]
    dmd_modes_split = np.split(dmd_modes, np.cumsum(var_lengths)[:-1], axis=1)
    dmd_modes = []
    for dmd_mode, orig_shape in zip(dmd_modes_split, orig_shapes):
        dmd_modes.append(dmd_mode.reshape((-1,) + orig_shape))
    return eigval, dmd_modes, dmd_mode_amplitudes

def POD(X, weights=None, rank=10):
    """
    Computes the POD using the method of snapshots.
    
    Parameters
    ----------
    X: List of snapshot matrices. Each element is a snapshot matrix for a physical variable.
       Each matrix can be multidimensional, but time must be the last axis.
    weight: List of weight matrices for weighting the snapshots. Each element conatins
       to the weight matrix for the corresponding element of X.
    Returns
    -------
    pod_modes: List of arrays of pod modes. Each element of the list gives the pod modes 
       for the corresponding physical variable given in X. Each pod mode array has layout 
       (mode_index, variable, space).
    eigval: eigenvalues corresponding to the pod_modes.
    pod_mode_amplitudes: Temporal amplitudes corresponding to the pod_modes
    """
    # Create the snapshot matrix
    orig_shapes = []
    var_lengths = []
    A = []
    for var in X:
        # Store the spatial shape
        orig_shapes.append(var.shape[:-1])
        # Store the spatial degrees of freedom
        var_lengths.append(np.prod(orig_shapes[-1]))
        # Flatten and concatenate variables
        if var.ndim != 2:
            # Must flatten spatial dimensions before SVD
            A.append(var.reshape((var_lengths[-1], -1)))
        else:
            A.append(var)
    # Convert to numpy array
    A = np.vstack(A)
    # Form the weight matrix
    if weights==None:
        W = sp.identity(A.shape[0])
    else:
        W = sp.block_diag(weights)
    # Form the covariance matrix
    Q = A.T @ W @ A
    # Perform the eigenvalue decompositions
    eigval, eigvec = sp.linalg.eigs(Q, k=rank)
    # Reconstruct the POD modes
    pod_modes = A @ eigvec @ np.diag(1/np.sqrt(eigval))
    # Make mode_index the first dimension
    pod_modes = pod_modes.T
    # Project to get mode_amplitudes
    pod_mode_amplitudes = pod_modes @ W @ A
    # Restructure the POD modes for output
    pod_modes_split = np.split(pod_modes, np.cumsum(var_lengths)[:-1], axis=1)
    pod_modes = []
    for pod_mode, orig_shape in zip(pod_modes_split, orig_shapes):
        pod_modes.append(pod_mode.reshape((-1,) + orig_shape))
    return eigval, pod_modes, pod_mode_amplitudes

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
    cb = ax.pcolormesh(x, y, cylinder_data, cmap=cmap, shading='gouraud')
    ax.set_xlim([-1, 8])
    ax.set_ylim([-2, 2])
    ax.add_patch(circle)
    ax.set_aspect(True)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cb, cax=cax)
    return cb

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