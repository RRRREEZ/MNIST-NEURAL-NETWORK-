import gzip
import numpy as np
import random
import tkinter as tk

def load_images(filename):

    with gzip.open(filename,'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(),dtype=np.uint8)

    return data.reshape(-1,28,28)


def load_labels(filename):

    with gzip.open(filename,'rb') as f:
        f.read(8)
        data = np.frombuffer(f.read(),dtype=np.uint8)

    return data

print("Loading MNIST dataset...")

dataset_loaded = False
images = None
labels = None

try:
    images = load_images("train-images-idx3-ubyte.gz")
    labels = load_labels("train-labels-idx1-ubyte.gz")
    dataset_loaded = True

    print("Dataset loaded:", len(images), "images")

except:
    print("Dataset not found. Training disabled.")

class Network:

    def __init__(self):

        self.w1 = np.random.randn(784,64) * 0.01
        self.w2 = np.random.randn(64,64) * 0.01
        self.w3 = np.random.randn(64,10) * 0.01

        self.b1 = np.zeros(64)
        self.b2 = np.zeros(64)
        self.b3 = np.zeros(10)

        self.lr = 0.01

    def relu(self,x):
        return np.maximum(0,x)

    def softmax(self,x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e, keepdims=True)

    def forward(self,x):

        self.z1 = x @ self.w1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = self.a2 @ self.w3 + self.b3
        self.out = self.softmax(self.z3)

        return self.out

    def predict(self,x):
        return np.argmax(self.forward(x))

    def train(self,x,y):

        out = self.forward(x)

        target = np.zeros(10)
        target[y] = 1

        error = out - target

        dw3 = np.outer(self.a2,error)
        db3 = error

        d2 = error @ self.w3.T
        d2[self.a2 <= 0] = 0

        dw2 = np.outer(self.a1, d2)
        db2 = d2

        d1 = d2 @ self.w2.T
        d1[self.a1 <= 0] = 0

        dw1 = np.outer(x, d1)
        db1 = d1

        self.w3 -= self.lr*dw3
        self.w2 -= self.lr*dw2
        self.w1 -= self.lr*dw1

        self.b3 -= self.lr*db3
        self.b2 -= self.lr*db2
        self.b1 -= self.lr*db1

class Trainer:

    def __init__(self,net,images,labels):

        self.net = net
        self.images = images
        self.labels = labels

        self.session_correct = 0
        self.session_total = 0

        self.global_correct = 0
        self.global_total = 0

    def train_step(self):

        i = random.randrange(len(self.images))

        x = self.images[i].reshape(784) / 255.0
        y = self.labels[i]

        pred = self.net.predict(x)

        self.net.train(x,y)

        self.session_total += 1
        self.global_total += 1

        if pred == y:
            self.session_correct += 1
            self.global_correct += 1

    def session_accuracy(self):
        if self.session_total==0:
            return 0
        return self.session_correct/self.session_total
    
    def reset_session(self):

        self.session_correct = 0
        self.session_total = 0

def save_model(net):

    np.savez("model.npz",
        w1=net.w1,
        w2=net.w2,
        w3=net.w3,
        b1=net.b1,
        b2=net.b2,
        b3=net.b3)

    print("model saved")

def load_model(net):

    data = np.load("model.npz")

    net.w1 = data["w1"]
    net.w2 = data["w2"]
    net.w3 = data["w3"]

    net.b1 = data["b1"] if "b1" in data else np.zeros(64)
    net.b2 = data["b2"] if "b2" in data else np.zeros(64)
    net.b3 = data["b3"] if "b3" in data else np.zeros(10)

    print("model loaded")

def gaussian_blur_numpy(img, sigma=1.0):

    size = 5
    k = size // 2
    y, x = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float32)
    padded = np.pad(img.astype(np.float32), k, mode='constant', constant_values=0)

    for i in range(h):
        for j in range(w):
            out[i, j] = np.sum(padded[i:i+size, j:j+size] * kernel)

    return out

def resize_numpy(img, new_h, new_w):

    old_h, old_w = img.shape
    out = np.zeros((new_h, new_w), dtype=np.float32)

    row_scale = old_h / new_h
    col_scale = old_w / new_w

    for i in range(new_h):
        for j in range(new_w):

            src_y = i * row_scale
            src_x = j * col_scale

            y0 = int(src_y)
            x0 = int(src_x)
            y1 = min(y0 + 1, old_h - 1)
            x1 = min(x0 + 1, old_w - 1)

            dy = src_y - y0
            dx = src_x - x0

            out[i, j] = (img[y0, x0] * (1 - dy) * (1 - dx) +
                         img[y1, x0] * dy       * (1 - dx) +
                         img[y0, x1] * (1 - dy) * dx       +
                         img[y1, x1] * dy       * dx)

    return out

def mnist_convert(grid):

    blurred = gaussian_blur_numpy(grid, sigma=1.0)

    rows = np.any(blurred > 10, axis=1)
    cols = np.any(blurred > 10, axis=0)

    if not rows.any() or not cols.any():
        return blurred.reshape(784) / 255.0

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = blurred[rmin:rmax+1, cmin:cmax+1]

    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = resize_numpy(cropped, new_h, new_w)

    canvas = np.zeros((28, 28), dtype=np.float32)
    pad_y = (28 - new_h) // 2
    pad_x = (28 - new_w) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    total = canvas.sum()
    if total > 0:
        ys, xs = np.mgrid[0:28, 0:28]
        cy = (ys * canvas).sum() / total
        cx = (xs * canvas).sum() / total

        shift_y = int(round(14 - cy))
        shift_x = int(round(14 - cx))

        canvas = np.roll(canvas, shift_y, axis=0)
        canvas = np.roll(canvas, shift_x, axis=1)

        if shift_y > 0:
            canvas[:shift_y, :] = 0
        elif shift_y < 0:
            canvas[shift_y:, :] = 0

        if shift_x > 0:
            canvas[:, :shift_x] = 0
        elif shift_x < 0:
            canvas[:, shift_x:] = 0

    return np.clip(canvas, 0, 255).reshape(784) / 255.0

class DrawWindow:

    def __init__(self,net):

        self.net = net

        self.grid = np.zeros((28,28))

        self.root = tk.Tk()

        self.canvas = tk.Canvas(self.root,width=280,height=280,bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>",self.draw)

        btn = tk.Button(self.root,text="Predict",command=self.predict)
        btn.pack()

        clear = tk.Button(self.root,text="Clear",command=self.clear)
        clear.pack()

    def draw(self,event):

        x=event.x
        y=event.y

        self.canvas.create_oval(x-8,y-8,x+8,y+8,fill="black")

        gx=int(x/10)
        gy=int(y/10)

        if gx<28 and gy<28:
            self.grid[gy][gx]=255

    def predict(self):

        x = mnist_convert(self.grid)

        p=self.net.predict(x)

        print("Prediction:",p)

    def clear(self):

        self.canvas.delete("all")
        self.grid[:]=0

    def run(self):
        self.root.mainloop()


    def update(self,x):

        self.canvas.delete("all")

        self.net.forward(x)

        self.draw_layer(self.net.a1,200)
        self.draw_layer(self.net.a2,350)
        self.draw_layer(self.net.out,500)

        self.root.update()

net = Network()
if dataset_loaded:
    trainer = Trainer(net, images, labels)
else:
    trainer = None

while True:

    cmd = input("ML> ").strip().lower()
    parts = cmd.split()

    if not parts:
        continue

    command = parts[0]

    if command == "train":

        if dataset_loaded:

            iterations = 1

            if len(parts) > 1:
                try:
                    iterations = int(parts[1])
                except:
                    print("Invalid number")

            print("Training for", iterations, "steps")

            for i in range(iterations):

                trainer.train_step()

                if i % 100 == 0:
                    print("step", i)

            print("Session accuracy:", trainer.session_accuracy())

        else:
            print("Dataset not found. Cannot train.")
            
    elif command == "draw":

        gui = DrawWindow(net)
        gui.run()

    elif command == "save":

        save_model(net)

    elif command == "load":

        load_model(net)

    elif command == "quit":

        break

    else:
        print("Unknown command")
