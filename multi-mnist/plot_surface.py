"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""
import argparse
import copy
import pickle
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
from plot import dataloader
from plot import evaluation
from plot import projection as proj
from plot import net_plotter
from plot import plot_2D
from plot import plot_1D
from plot import model_loader
from plot import scheduler
from plot import mpi4pytorch as mpi

from model_lenet import RegressionModel


def name_surface_file(args, dir_file):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += "_[%s,%s,%d]" % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += "x[%s,%s,%d]" % (str(args.ymin), str(args.ymax), int(args.ynum))

    # dataloder parameters
    if args.raw_data:  # without data normalization
        surf_file += "_rawdata"
    if args.data_split > 1:
        surf_file += (
            "_datasplit=" + str(args.data_split) + "_splitidx=" + str(args.split_idx)
        )

    return surf_file + ".h5"


def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, "r")
        if (args.y and "ycoordinates" in f.keys()) or "xcoordinates" in f.keys():
            f.close()
            print("%s is already set up" % surf_file)
            return

    f = h5py.File(surf_file, "a")
    f["dir_file"] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=args.xnum)
    f["xcoordinates"] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=args.ynum)
        f["ycoordinates"] = ycoordinates
    f.close()

    return surf_file


def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, comm, rank, args):
    """
    Calculate the loss values and accuracies of modified models in parallel
    using MPI reduce.
    """

    f = h5py.File(surf_file, "r+" if rank == 0 else "r")
    losses, accuracies = [], []
    xcoordinates = f["xcoordinates"][:]
    ycoordinates = f["ycoordinates"][:] if "ycoordinates" in f.keys() else None

    if loss_key not in f.keys():
        shape = (
            xcoordinates.shape
            if ycoordinates is None
            else (len(xcoordinates), len(ycoordinates))
        )
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(
        losses, xcoordinates, ycoordinates, comm
    )

    print("Computing %d values for rank %d" % (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    criterion = nn.CrossEntropyLoss()

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == "weights":
            net_plotter.set_weights(net, w, d, coord)
        elif args.dir_type == "states":
            net_plotter.set_states(net, s, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()

        net.eval()

        acc_1 = 0
        acc_2 = 0
        loss_1 = 0
        loss_2 = 0
        with torch.no_grad():

            for (it, batch) in enumerate(dataloader):
                X = batch[0]
                y = batch[1]
                X = X.cuda()
                y = y.cuda()

                out1_prob, out2_prob = net(X)

                loss1 = criterion(out1_prob, y[:, 0])
                loss2 = criterion(out2_prob, y[:, 1])

                out1 = out1_prob.max(1)[1]
                out2 = out2_prob.max(1)[1]
                acc_1 += (out1 == y[:, 0]).sum()
                acc_2 += (out2 == y[:, 1]).sum()
                loss_1 += loss1.item()
                loss_2 += loss2.item()
            acc_1 = acc_1.item() / len(dataloader.dataset)
            acc_2 = acc_2.item() / len(dataloader.dataset)
            loss_1 = loss_1 / len(dataloader.dataset)
            loss_2 = loss_2 / len(dataloader.dataset)
        # loss, acc = evaluation.eval_loss(net, criterion, dataloader, args.cuda)

        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f.flush()

        print(
            "Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f"
            % (
                rank,
                count,
                len(inds),
                100.0 * count / len(inds),
                str(coord),
                loss_key,
                loss,
                acc_key,
                acc,
                loss_compute_time,
                syc_time,
            )
        )

    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)

    total_time = time.time() - start_time
    print("Rank %d done!  Total time: %.2f Sync: %.2f" % (rank, total_time, total_sync))

    f.close()


###############################################################
#                          MAIN
###############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plotting loss surface")
    parser.add_argument("--mpi", "-m", action="store_true", help="use mpi")
    parser.add_argument("--cuda", "-c", action="store_true", help="use cuda")
    parser.add_argument("--threads", default=2, type=int, help="number of threads")
    parser.add_argument(
        "--ngpu",
        type=int,
        default=1,
        help="number of GPUs to use for each rank, useful for data parallel evaluation",
    )
    parser.add_argument("--batch_size", default=128, type=int, help="minibatch size")

    # model parameters
    parser.add_argument(
        "--dset",
        default="multi_fashion_and_mnist",
        type=str,
        help="Dataset for training.",
    )

    parser.add_argument(
        "--model_file", default="", help="path to the trained model file"
    )
    parser.add_argument(
        "--loss_name",
        "-l",
        default="crossentropy",
        help="loss functions: crossentropy | mse",
    )

    # direction parameters
    parser.add_argument(
        "--dir_file",
        default="",
        help="specify the name of direction file, or the path to an eisting direction file",
    )
    parser.add_argument(
        "--dir_type",
        default="weights",
        help="direction type: weights | states (including BN's running_mean/var)",
    )
    parser.add_argument(
        "--x", default="-1:1:51", help="A string with format xmin:x_max:xnum"
    )
    parser.add_argument("--y", default=None, help="A string with format ymin:ymax:ynum")
    parser.add_argument(
        "--xnorm", default="", help="direction normalization: filter | layer | weight"
    )
    parser.add_argument(
        "--ynorm", default="", help="direction normalization: filter | layer | weight"
    )
    parser.add_argument(
        "--xignore", default="", help="ignore bias and BN parameters: biasbn"
    )
    parser.add_argument(
        "--yignore", default="", help="ignore bias and BN parameters: biasbn"
    )
    parser.add_argument(
        "--same_dir",
        action="store_true",
        default=False,
        help="use the same random direction for both x-axis and y-axis",
    )
    parser.add_argument(
        "--idx", default=0, type=int, help="the index for the repeatness experiment"
    )
    parser.add_argument(
        "--surf_file",
        default="",
        help="customize the name of surface file, could be an existing file.",
    )

    # plot parameters
    parser.add_argument(
        "--proj_file",
        default="",
        help="the .h5 file contains projected optimization trajectory.",
    )
    parser.add_argument(
        "--loss_max", default=5, type=float, help="Maximum value to show in 1D plot"
    )
    parser.add_argument("--vmax", default=10, type=float, help="Maximum value to map")
    parser.add_argument("--vmin", default=0.1, type=float, help="Miminum value to map")
    parser.add_argument(
        "--vlevel", default=0.5, type=float, help="plot contours every vlevel"
    )
    parser.add_argument(
        "--show", action="store_true", default=False, help="show plotted figures"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="use log scale for loss values",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="plot figures after computation",
    )

    args = parser.parse_args()

    torch.manual_seed(123)
    # --------------------------------------------------------------------------
    # Environment setup
    # --------------------------------------------------------------------------
    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception(
                "User selected cuda option, but cuda is not available on this machine"
            )
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print(
            "Rank %d use GPU %d of %d GPUs on %s"
            % (rank, torch.cuda.current_device(), gpu_count, socket.gethostname())
        )

    # --------------------------------------------------------------------------
    # Check plotting resolution
    # --------------------------------------------------------------------------
    try:
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(":")]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(":")]
            assert (
                args.ymin and args.ymax and args.ynum
            ), "You specified some arguments for the y axis, but not all"
    except:
        raise Exception(
            "Improper format for x- or y-coordinates. Try something like -1:1:51"
        )

    # --------------------------------------------------------------------------
    # Load models and extract parameters
    # --------------------------------------------------------------------------
    net = RegressionModel(2).cuda()
    w = net_plotter.get_weights(net)  # initial parameters
    s = copy.deepcopy(net.state_dict())  # deepcopy since state_dict are references

    # --------------------------------------------------------------------------
    # Setup the direction file and the surface file
    # --------------------------------------------------------------------------
    dir_file = net_plotter.name_direction_file(args)  # name the direction file
    if rank == 0:
        net_plotter.setup_direction(args, dir_file, net)

    surf_file = name_surface_file(args, dir_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file)

    # wait until master has setup the direction file and surface file
    mpi.barrier(comm)

    # load directions
    d = net_plotter.load_directions(dir_file)
    # calculate the consine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(
            proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1])
        )
        print("cosine similarity between x-axis and y-axis: %f" % similarity)

    mpi.barrier(comm)

    with open(f"./data/{args.dset}.pickle", "rb") as f:
        trainX, trainLabel, testX, testLabel = pickle.load(f)
    trainX = torch.from_numpy(trainX.reshape(120000, 1, 36, 36)).float()
    trainLabel = torch.from_numpy(trainLabel).long()
    testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
    testLabel = torch.from_numpy(testLabel).long()
    train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
    test_set = torch.utils.data.TensorDataset(testX, testLabel)

    trainloader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False
    )

    # --------------------------------------------------------------------------
    # Start the computation
    # --------------------------------------------------------------------------
    crunch(
        surf_file,
        net,
        w,
        s,
        d,
        trainloader,
        "train_loss",
        "train_acc",
        comm,
        rank,
        args,
    )
    # crunch(surf_file, net, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, args)

    # --------------------------------------------------------------------------
    # Plot figures
    # --------------------------------------------------------------------------
    if args.plot and rank == 0:
        if args.y and args.proj_file:
            plot_2D.plot_contour_trajectory(
                surf_file, dir_file, args.proj_file, "train_loss", args.show
            )
        elif args.y:
            plot_2D.plot_2d_contour(
                surf_file, "train_loss", args.vmin, args.vmax, args.vlevel, args.show
            )
        else:
            plot_1D.plot_1d_loss_err(
                surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show
            )
