# MNIST-WGAN
<b>Overview</b><br>
This project generates hand-written digits from the MNIST dataset using WGAN architecture.
<br><b>Use</b><br>
Before use, one should verify that the network architecture is as-desired. This may be done with the GUI to the right of the number display. The "Default" button resets the network to hard-coded values which I have verified function. The "Reset" button sets the ACTIVE network to whatever architecture is displayed. Also, resetting ONLY DOES SO FOR THE DISPLAYED NETWORK, which may be unintuitive. In order to change which network is displayed, use the "Critic \[1\] or Generator \[0\]" checkbox.
<br>To begin training the network, press the "Train" button, after which you may use the "Clear" button to reset the average error and average percent correct value textboxes.
<br><b>Documentation</b><br>
The back-end can be kind-of confusing, but it's fairly well-written and documented. While the front-end is less clear, it's mostly just form-fitting CSV lists to generate whatever architecture is desired. I find it easier to modify the default values (lines 395-447) and reset the network from there, but this isn't really necessary (it just saves a bit of time). The GUI is probably easier if you don't plan to spend lots of time modifying this to suit your purposes.
<br><b>Modification</b><br>
In order to use the project on an alternative dataset, one must replace the MNIST files, then rewrite IO.FindNextNumber (and corresponding file paths) to read the desired file. Nothing else stops this from being done. It would certainly be easier than doing this from-scratch again.
<br><b>Future plans</b><br>
There's still some improvements I could make to the project (like letting the front-end assign no activation function to a layer). However, I've been working on this for so long that I feel like letting this rest as-is for a while.
<br><br>
<b>Example</b><br><br>
![A 1](https://github.com/qdm097/MNIST-WGAN/blob/master/WGAN1/WGAN8.PNG)
<br><br>

