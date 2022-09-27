import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import glob, os
mycolors = sns.color_palette("RdBu", 10)
deeppallet = sns.color_palette("deep")

def points_on_circle(radius, imageSizeX, imageSizeY, x0=0, y0=0):
    radius = int(radius*imageSizeX)
    x0, y0 = int(x0*imageSizeX), int(y0*imageSizeY)
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 == radius**2) #index
    for x_out, y_out in zip(x_[x], y_[y]):
        if x_out>0 and y_out>0 and x_out<imageSizeX and y_out<imageSizeY:
            yield x_out/imageSizeX, y_out/imageSizeY


def sns_setup(sns):
    sns.set(rc={'figure.figsize':(12,8)})
    sns.set_context("paper", rc={"font.size":30,"axes.titlesize":25,"axes.labelsize":20,
                                 "legend.title_fontsize":30,"legend.fontsize":25,
                                 "xtick.labelsize":25, "ytick.labelsize":25,
                                 'legend.frameon': False})
    sns.set_style("white")
    sns.set_palette("deep")
    return

def sns_setup_small(sns):
    sns.set(rc={'figure.figsize':(12,8)})
    sns.set_context("paper", rc={"font.size":20,"axes.titlesize":24,"axes.labelsize":20,
                                 "legend.title_fontsize":20,"legend.fontsize":15,
                                 "xtick.labelsize":20, "ytick.labelsize":20})
    sns.set_style("white")
    sns.set_palette("deep")
    return

def label_correct(row):
    if row['gt']==row['resp']:
        return 1
    else:
        return 0

def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def get_pdf(x,mean,cov):
    return multivariate_normal.pdf(x, mean=mean, cov=cov)
def get_cdf(x,mean,cov):
    return multivariate_normal.cdf(x, mean=mean, cov=cov)
def dot_sum(x1,x2):
    return (x1*x2).sum()

def get_stimuli(lth, wth, ang, imageSizeX, imageSizeY, sigma=.08):
    x = np.linspace(0, imageSizeX-1, imageSizeX).astype(int)
    y = np.linspace(0, imageSizeX-1, imageSizeY).astype(int)
    columnsInImage, rowsInImage = np.meshgrid(x, y)
    centerX = int((imageSizeX+1)/2)
    centerY = int((imageSizeY+1)/2)

    b = lth
    a = wth
    theta = (90-ang)*np.pi/180

    img = ( ( (columnsInImage - centerX)*np.cos(theta)+(rowsInImage - centerY)*np.sin(theta) )**2/a**2 + \
            ( ( columnsInImage - centerX)*np.sin(theta)-(rowsInImage - centerY)*np.cos(theta) )**2/b**2 <= 1).astype(float)\
          *140/255
    img[img==0]=.5
    stimuli = img + sigma*np.random.randn(img.shape[0],img.shape[1])
    return img, stimuli

def get_stimuliall(lth, wth, ang, dis, r, imageSizeX, imageSizeY, sigma=.08):
    x = np.linspace(0, imageSizeX-1, imageSizeX).astype(int)
    y = np.linspace(0, imageSizeX-1, imageSizeY).astype(int)
    columnsInImage, rowsInImage = np.meshgrid(x, y)
    centerX = int((imageSizeX+1)/2)
    centerY = int((imageSizeY+1)/2)

    # ellipse (for a specific shape)
    b = lth
    a = wth
    theta = (90-ang)*np.pi/180
    img = ( ( (columnsInImage - centerX)*np.cos(theta)+(rowsInImage - centerY)*np.sin(theta) )**2/a**2 + \
            ( ( columnsInImage - centerX)*np.sin(theta)-(rowsInImage - centerY)*np.cos(theta) )**2/b**2 <= 1).astype(float)\
          *140/255

    # triangle (for the same distance, multiple locations)

    # circle (for a specific radius)


    img[img==0]=.5
    stimuli = img + sigma*np.random.randn(img.shape[0],img.shape[1])
    return img, stimuli

def setupsignal(imageSizeX, imageSizeY, means, covs, feature_range):
    # this function vectorize signal images and calculate joint probabilities based on multivariate guassian distribution
    img_yrang,img_xrang = [0,imageSizeX], [0,imageSizeY]
    n_class = feature_range.shape[0]
    prior = 1/n_class #default same prior

    # set up unique signals maxtrix for each class (#signals x #pixels)
    for c in range(n_class):
        signal_temp = []  # each column is vectorized signal
        signal_p = []  # probabilty of observing each unique signal in the class
        # for each unique sample in a class
        for x1 in np.arange(feature_range[c][0][0], feature_range[c][0][1] + 1):
            for x2 in np.arange(feature_range[c][1][0], feature_range[c][1][1] + 1):
                for x3 in np.arange(feature_range[c][2][0], feature_range[c][2][1] + 1):
                    # for x4 in np.arange(feature_range[c][3][0], feature_range[c][3][1] + 1):
                    #     for x5 in np.arange(feature_range[c][4][0], feature_range[c][4][1] + 1):
                    # print('x1:{}, x2:{}, x3:{}'.format(x1, x2, x3))
                    p_sk = get_pdf([x1, x2, x3], means[c], covs[c])
                    #joint probability for known ellipse shape, triangle distance, circle radius
                    signal_p.append(p_sk)
                    # for the same triangle distance, there will be many locations
                    img, img_n = get_stimuli(x1, x2, x3, imageSizeX, imageSizeY)
                    # get just center part to reduce computations
                    img_crop = img[img_yrang[0]:img_yrang[1], img_xrang[0]:img_xrang[1]]
                    signal_temp.append(np.reshape(img_crop, (1, -1)).tolist())  # add vectorized signal
        signal_temp = np.array(signal_temp).squeeze(1)
        signal_p = np.array(signal_p)
        np.save('IO_data/class{}_signal.npy'.format(c + 1), signal_temp)
        np.save('IO_data/class{}_signal_p.npy'.format(c + 1), signal_p)


def prop_resp(resp):
    features = ['length','width','angle']
    for i in range(len(features)):
        f = features[i]
        ptrials = resp.groupby(f).mean().reset_index() #proportion of trials saying yes vs no for one feature
    return ptrials

def plot_tuningcurve(resp, filename):
    sns_setup(sns)
    fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(12, 8))

    features = ['length','width','angle']
    for i in range(len(features)):
        f = features[i]
        ptrials = resp.groupby(f).mean().reset_index() #proportion of trials saying yes vs no for one feature
        sns.lineplot(x=ptrials[f], y=ptrials['resp'], ax=axes[0,i])
        sns.lineplot(x=ptrials[f], y=ptrials['gt'], ax=axes[1, i])
        if i==0:
            axes[0,i].set(ylabel='%responding "target"')
            axes[1,i].set(ylabel='%target')
        else:
            axes[0,i].set(ylabel='')
            axes[1,i].set(ylabel='')
    plt.tight_layout()
    plt.savefig('Figures/tuningcurve_{}.jpg'.format(filename))
    plt.close(fig)


def organize_humanresp(data_path, subject):
    stim_info = pd.read_excel('Stimuli/stim_info.xlsx')
    files = glob.glob('{}/{}/*'.format(data_path, subject))
    num_block = len(files)
    print('subject {} has {} blocks'.format(subject, num_block))
    allresp = pd.DataFrame()
    # b = 1
    f_idx = 0
    for f in files:
        resp = pd.read_excel(f)
        length, width, angle, trian_dis, circle_r = [],[],[],[],[]
        for r in resp.iterrows():
            trial = stim_info[stim_info['stim'] == os.path.split(r[1][0])[-1][:-4]]
            if len(trial)==0:
                trial = stim_info[stim_info['stim'] == r[1][0].split('\\')[-1][:-4]]
            length.append(trial['length'].item())
            width.append(trial['width'].item())
            angle.append(trial['angle'].item())
            trian_dis.append(trial['trian_dis'].item()) #target triangle lower x position (lower value)
            circle_r.append(trial['circle_r'].item())  #target circle lower y position (higher value)
        resp['length'] = length
        resp['width'] = width
        resp['angle'] = angle
        resp['trian_dis'] = trian_dis #triangle_distance
        resp['circle_r'] = circle_r #circle radius
        # resp['block_num'] = r[0]//200+1 #200 trials per block
        allresp = allresp.append(resp, ignore_index=True)
    allresp['block_num'] = allresp.index//200+1 #200 trials per block
    allresp = allresp.rename(columns = {'corr_ans':'gt'})
    return allresp