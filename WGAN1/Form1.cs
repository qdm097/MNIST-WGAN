using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading;
using System.Windows.Forms;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace WGAN1
{
    public partial class Form1 : Form
    {
        double learningrate = 0.0000146;
        double rmsdecay = 0.7;
        double clippingparameter = 1;
        int batchsize = 5;
        int ctogratio = 5;
        int generatorlcount = 5;
        int gncount = 28;
        int criticlcount = 5;
        int cncount = 30;
        bool dt;
        public bool DoneTraining { get { return dt; } set { dt = value; if (dt) { TrainBtn.Enabled = true; dt = false; } } }
        string cs;
        public string CScore { get { return cs; } set { cs = value; CScoreTxt.Text = value; } }
        int[,] img;
        public int[,] image {
            get { return img; }
            set { 
                img = value;
                NumPB.Image = FromTwoDimIntArrayGray(Scaler(img, 10));
            } 
        }
        public Form1()
        {
            InitializeComponent();
            ClipTxt.Text = clippingparameter.ToString();
            AlphaTxt.Text = learningrate.ToString();
            RMSDTxt.Text = rmsdecay.ToString();
            MTxt.Text = batchsize.ToString();
            CTGTxt.Text = ctogratio.ToString();
        }
        private void TrainBtn_Click(object sender, EventArgs e)
        {
            if (NN.Training) { NN.Training = false; TrainBtn.Enabled = false; return; }
            NN.Training = true;
            var thread = new Thread(() => 
            {
                NN.Train(true, criticlcount, generatorlcount, GenerateCounts(criticlcount, cncount, true),
                    GenerateCounts(generatorlcount, gncount, false), GenerateTypes(generatorlcount, false), 
                    GenerateTypes(criticlcount, true), 28, learningrate, clippingparameter, batchsize, ctogratio,
                    rmsdecay, 7, 28, this, 0);               
            });
            thread.IsBackground = true;
            thread.Start();
        }
        /// <summary>
        /// For now I'm manually setting the conv layer points
        /// </summary>
        /// <param name="numlayers">How many layers are in the network</param>
        /// <returns></returns>
        List<iLayer> GenerateTypes(int numlayers, bool cog)
        {
            var list = new List<iLayer>();
            for (int i = 0; i < numlayers; i++)
            {
                //Layers don't have to be inititalized because they're just a list of types
                if (!cog && i == 3) { list.Add(new ConvolutionLayer(0, 0)); }
                else { list.Add(new FullyConnectedLayer(0, 0)); }
            }
            return list;
        }
        private List<int> GenerateCounts(int layercount, int nperlayer, bool cog)
        {
            var output = new List<int>();
            for (int i = 0; i < layercount; i++)
            {
                //Critic output layer has 1 neuron
                if (cog && i == layercount - 1) { output.Add(1); continue; }
                //Manually setting the c-layer's position and kernel size for now
                if (i == 3) { output.Add(100); continue; }
                //Generator layer has 28 * 28 neurons (if not a c-layer)
                if (!cog && i == layercount - 1) { output.Add(28 * 28); continue; }
                output.Add(nperlayer); 
            }

            return output;
        }

        private void AlphaTxt_TextChanged(object sender, EventArgs e)
        {
            if (!double.TryParse(AlphaTxt.Text, out double lr)) { MessageBox.Show("NAN"); return; }
            if (lr < 0 || lr > 1) { MessageBox.Show("Learning rate must be between 0 and 1"); return; }
            learningrate = lr;
        }

        private void RMSDTxt_TextChanged(object sender, EventArgs e)
        {
            if (!double.TryParse(RMSDTxt.Text, out double rmsrate)) { MessageBox.Show("NAN"); return; }
            if (rmsrate < 0 || rmsrate > 1) { MessageBox.Show("Invalid RMS decay rate"); return; }
            rmsdecay = rmsrate;
        }

        private void MTxt_TextChanged(object sender, EventArgs e)
        {
            if (!int.TryParse(MTxt.Text, out int bs)) { MessageBox.Show("NAN"); return; }
            if (bs < 0 || bs > 1000) { MessageBox.Show("Batch size must be between 0 and 1000"); return; }
            batchsize = bs;
        }

        private void CTGTxt_TextChanged(object sender, EventArgs e)
        {
            if (!int.TryParse(CTGTxt.Text, out int ctgr)) { MessageBox.Show("NAN"); return; }
            if (ctgr < 1 || ctgr > 50) { MessageBox.Show("The critic to generator ratio must be between 1 and 50"); return; }
            ctogratio = ctgr;
        }
        private void ClipTxt_TextChanged(object sender, EventArgs e)
        {
            if (!int.TryParse(ClipTxt.Text, out int clippar)) { MessageBox.Show("NAN"); return; }
            if (clippar <= 0 || clippar > 10) { MessageBox.Show("The clipping parameter must be between 0 and 10"); return; }
            clippingparameter = clippar;
        }
        private void ClearBtn_Click(object sender, EventArgs e)
        {
            NN.Clear = true;
        }
        public int[,] Scaler(int[,] input, int scale)
        {
            int[,] scaled = new int[28 * scale, 28 * scale];
            //Foreach int in Obstacles
            for (int j = 0; j < 28; j++)
            {
                for (int jj = 0; jj < 28; jj++)
                {
                    //Scale by scale
                    for (int i = 0; i < scale; i++)
                    {
                        for (int ii = 0; ii < scale; ii++)
                        {
                            scaled[(j * scale) + i, (jj * scale) + ii] = input[jj, j];
                        }
                    }
                }
            }
            return scaled;
        }
        public static Bitmap FromTwoDimIntArrayGray(Int32[,] data)
        {
            // Transform 2-dimensional Int32 array to 1-byte-per-pixel byte array
            Int32 width = data.GetLength(0);
            Int32 height = data.GetLength(1);
            Int32 byteIndex = 0;
            Byte[] dataBytes = new Byte[height * width];
            for (Int32 y = 0; y < height; y++)
            {
                for (Int32 x = 0; x < width; x++)
                {
                    // logical AND to be 100% sure the int32 value fits inside
                    // the byte even if it contains more data (like, full ARGB).
                    dataBytes[byteIndex] = (Byte)(((UInt32)data[x, y]) & 0xFF);
                    // More efficient than multiplying
                    byteIndex++;
                }
            }
            // generate palette
            Color[] palette = new Color[256];
            for (Int32 b = 0; b < 256; b++)
                palette[b] = Color.FromArgb(b, b, b);
            // Build image
            return BuildImage(dataBytes, width, height, width, PixelFormat.Format8bppIndexed, palette, null);
        }
        /// <summary>
        /// Creates a bitmap based on data, width, height, stride and pixel format.
        /// </summary>
        /// <param name="sourceData">Byte array of raw source data</param>
        /// <param name="width">Width of the image</param>
        /// <param name="height">Height of the image</param>
        /// <param name="stride">Scanline length inside the data</param>
        /// <param name="pixelFormat">Pixel format</param>
        /// <param name="palette">Color palette</param>
        /// <param name="defaultColor">Default color to fill in on the palette if the given colors don't fully fill it.</param>
        /// <returns>The new image</returns>
        public static Bitmap BuildImage(Byte[] sourceData, Int32 width, Int32 height, Int32 stride, PixelFormat pixelFormat, Color[] palette, Color? defaultColor)
        {
            Bitmap newImage = new Bitmap(width, height, pixelFormat);
            BitmapData targetData = newImage.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, newImage.PixelFormat);
            Int32 newDataWidth = ((Image.GetPixelFormatSize(pixelFormat) * width) + 7) / 8;
            // Compensate for possible negative stride on BMP format.
            Boolean isFlipped = stride < 0;
            stride = Math.Abs(stride);
            // Cache these to avoid unnecessary getter calls.
            Int32 targetStride = targetData.Stride;
            Int64 scan0 = targetData.Scan0.ToInt64();
            for (Int32 y = 0; y < height; y++)
                Marshal.Copy(sourceData, y * stride, new IntPtr(scan0 + y * targetStride), newDataWidth);
            newImage.UnlockBits(targetData);
            // Fix negative stride on BMP format.
            if (isFlipped)
                newImage.RotateFlip(RotateFlipType.Rotate180FlipX);
            // For indexed images, set the palette.
            if ((pixelFormat & PixelFormat.Indexed) != 0 && palette != null)
            {
                ColorPalette pal = newImage.Palette;
                for (Int32 i = 0; i < pal.Entries.Length; i++)
                {
                    if (i < palette.Length)
                        pal.Entries[i] = palette[i];
                    else if (defaultColor.HasValue)
                        pal.Entries[i] = defaultColor.Value;
                    else
                        break;
                }
                newImage.Palette = pal;
            }
            return newImage;
        }
    }
}
