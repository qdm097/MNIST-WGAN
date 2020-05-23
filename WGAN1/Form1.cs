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
        public List<string> LayerTypes { get; set; }
        public List<int> LayerCounts { get; set; }
        public List<string> InactiveLayerTypes { get; set; }
        public List<int> InactiveLayerCounts { get; set; }
        NN Critic;
        NN Generator;
        int batchsize = 5;
        int ctogratio = 5;
        int imgspeed = 0;

        int resolution = 28;
        int latentsize = 36;
        //The maximum RMSE allowed before the network stops learning
        public static int Cutoff = 10;
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
            //Layer types combobox
            LayerTypeCB.Items.Add("Fully Connected");
            LayerTypeCB.Items.Add("Convolution");
            LayerTypeCB.SelectedIndex = 0;

            ClipTxt.Text = NN.ClipParameter.ToString();
            AlphaTxt.Text = NN.LearningRate.ToString();
            RMSDTxt.Text = NN.RMSDecay.ToString();
            MTxt.Text = batchsize.ToString();
            CTGTxt.Text = ctogratio.ToString();
            try
            {
                Critic = IO.Read(true);
                Generator = IO.Read(false);
            }
            catch
            {
                ResetBtn_Click(this, new EventArgs());
            }
            RefreshListBoxes(Generator, true);
            RefreshListBoxes(Critic, false);
        }
        private void TrainBtn_Click(object sender, EventArgs e)
        {
            if (NN.Training) { NN.Training = false; TrainBtn.Enabled = false; return; }
            NN.Training = true;
            var thread = new Thread(() => 
            {
                NN.Train(Critic, Generator, latentsize, resolution, batchsize, ctogratio, 1, this, imgspeed);               
            });
            thread.IsBackground = true;
            thread.Start();
        }
        private void ResetBtn_Click(object sender, EventArgs e)
        {
            if (NN.Training) { NN.Save = false; NN.Training = false; TrainBtn.Enabled = false; }
            //Generator
            Generator = new NN().Init(GenerateLayers(false), false);
            IO.Write(Generator, false);
            //Critic
            Critic = new NN().Init(GenerateLayers(true), true);
            IO.Write(Critic, true);
        }
        List<iLayer> GenerateLayers(bool COG)
        {
            int priorsize;
            if (COG) { priorsize = resolution * resolution; }
            else { priorsize = latentsize; }
            List<int> layercounts;
            List<string> layertypes;
            if (this.COG.Checked == COG)
            {
                layercounts = LayerCounts;
                layertypes = LayerTypes;
            }
            else
            {
                layercounts = InactiveLayerCounts;
                layertypes = InactiveLayerTypes;
            }
            List<iLayer> layers = new List<iLayer>();
            for (int i = 0; i < layercounts.Count; i++)
            {
                int ncount = layercounts[i];
                //If a convolution
                if (layertypes[i] == "c")
                {
                    layers.Add(new ConvolutionLayer(ncount, priorsize));
                    (layers[i] as ConvolutionLayer).COG = COG;
                    //Calculate the padded matrix size (if applicable)
                    int temp = (int)Math.Sqrt(priorsize);
                    
                    if (COG)
                    {
                        //Critic decreases the size of the inputted array
                        priorsize = (int)((temp / ConvolutionLayer.StepSize) - ncount + 1);
                    }
                    else
                    {
                        //Generator increases the size of the inputted array
                        priorsize = (int)(temp + ncount - 1);
                    }
                    priorsize *= priorsize;
                    continue;
                }
                if (layertypes[i] == "f")
                {
                    layers.Add(new FullyConnectedLayer(ncount, priorsize));
                    priorsize = ncount;
                    continue;
                }
                throw new Exception("Invalid layer type");
            }
            return layers;
        }
        private void AlphaTxt_TextChanged(object sender, EventArgs e)
        {
            if (!double.TryParse(AlphaTxt.Text, out double lr)) { MessageBox.Show("NAN"); return; }
            if (lr < 0 || lr > 1) { MessageBox.Show("Learning rate must be between 0 and 1"); return; }
            NN.LearningRate = lr;
        }

        private void RMSDTxt_TextChanged(object sender, EventArgs e)
        {
            if (!double.TryParse(RMSDTxt.Text, out double rmsrate)) { MessageBox.Show("NAN"); return; }
            if (rmsrate < 0 || rmsrate > 1) { MessageBox.Show("Invalid RMS decay rate"); return; }
            NN.RMSDecay = rmsrate;
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
            if (!double.TryParse(ClipTxt.Text, out double clippar)) { MessageBox.Show("NAN"); return; }
            if (clippar <= 0 || clippar > 10) { MessageBox.Show("The clipping parameter must be between 0 and 10"); return; }
            NN.ClipParameter = clippar;
        }
        private void ClearBtn_Click(object sender, EventArgs e)
        {
            NN.Clear = true; CScore = null;
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
        private List<string> DefaultTypes()
        {
            var list = new List<string>();
            if (COG.Checked)
            {
                list.Add("c");
                list.Add("c");
                list.Add("f");
                list.Add("f");
                list.Add("f");
            }
            else
            {
                list.Add("f");
                list.Add("f");
                list.Add("c");
                list.Add("c");
                list.Add("f");
            }
            return list;
        }
        private List<int> DefaultCounts()
        {
            var list = new List<int>();
            if (COG.Checked)
            {
                list.Add(3);
                list.Add(2);
                list.Add(36);
                list.Add(17);
                list.Add(1);
            }
            else
            {
                list.Add(49);
                list.Add(100);
                list.Add(3);
                list.Add(2);
                list.Add(28 * 28);
            }
            return list;
        }
        private void RefreshListBoxes(NN desired, bool ActiveOrInactive)
        {
            var types = new List<string>();
            var counts = new List<int>();
            LayerLB.Items.Clear();

            foreach (iLayer l in desired.Layers)
            {
                string name = null;
                int len = l.Length;
                if (l is FullyConnectedLayer) { types.Add("f"); name = "Fully Connected"; }
                if (l is ConvolutionLayer) { types.Add("c"); name = "Convolution"; len = (l as ConvolutionLayer).KernelSize; }
                counts.Add(len);
                LayerLB.Items.Add("[" + (counts.Count - 1).ToString() + "] " + name + ", " + len.ToString());
            }
            if (ActiveOrInactive)
            {
                LayerTypes = types;
                LayerCounts = counts;
            }
            else
            {
                InactiveLayerTypes = types;
                InactiveLayerCounts = counts;
            }
        }
        private void LayerLB_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (LayerLB.SelectedIndex < 0) { return; }
            LayerCountTxt.Text = LayerCounts[LayerLB.SelectedIndex].ToString();
        }

        private void UpBtn_Click(object sender, EventArgs e)
        {
            if (LayerLB.Items.Count < 2) { return; }
            if (LayerLB.SelectedIndex == 0) { return; }

            //Layercounts
            int i = LayerCounts[LayerLB.SelectedIndex];
            LayerCounts[LayerLB.SelectedIndex] = LayerCounts[LayerLB.SelectedIndex - 1];
            LayerCounts[LayerLB.SelectedIndex - 1] = i;

            //Layer types
            string s = LayerTypes[LayerLB.SelectedIndex];
            LayerTypes[LayerLB.SelectedIndex] = LayerTypes[LayerLB.SelectedIndex - 1];
            LayerTypes[LayerLB.SelectedIndex - 1] = s;

            //Layer list box
            string selected = LayerLB.Items[LayerLB.SelectedIndex].ToString();
            string replaced = LayerLB.Items[LayerLB.SelectedIndex - 1].ToString();
            selected = selected.Remove(1, 1).Insert(1, (LayerLB.SelectedIndex - 1).ToString());
            replaced = replaced.Remove(1, 1).Insert(1, (LayerLB.SelectedIndex).ToString());
            LayerLB.Items[LayerLB.SelectedIndex] = replaced;
            LayerLB.Items[LayerLB.SelectedIndex - 1] = selected;
            LayerLB.SelectedIndex--;
        }

        private void DownBtn_Click(object sender, EventArgs e)
        {
            if (LayerLB.Items.Count < 2) { return; }
            if (LayerLB.SelectedIndex == LayerLB.Items.Count - 1) { return; }

            //Layercounts
            int i = LayerCounts[LayerLB.SelectedIndex];
            LayerCounts[LayerLB.SelectedIndex] = LayerCounts[LayerLB.SelectedIndex + 1];
            LayerCounts[LayerLB.SelectedIndex + 1] = i;

            //Layer types
            string s = LayerTypes[LayerLB.SelectedIndex];
            LayerTypes[LayerLB.SelectedIndex] = LayerTypes[LayerLB.SelectedIndex + 1];
            LayerTypes[LayerLB.SelectedIndex + 1] = s;

            //Layer list box
            string selected = LayerLB.Items[LayerLB.SelectedIndex].ToString();
            string replaced = LayerLB.Items[LayerLB.SelectedIndex + 1].ToString();
            selected = selected.Remove(1, 1).Insert(1, (LayerLB.SelectedIndex + 1).ToString());
            replaced = replaced.Remove(1, 1).Insert(1, (LayerLB.SelectedIndex).ToString());
            LayerLB.Items[LayerLB.SelectedIndex] = replaced;
            LayerLB.Items[LayerLB.SelectedIndex + 1] = selected;
            LayerLB.SelectedIndex++;
        }

        private void AddBtn_Click(object sender, EventArgs e)
        {
            int result = int.Parse(LayerCountTxt.Text);
            string type = null;
            if (LayerTypeCB.Text == "Fully Connected") { type = "f"; }
            if (LayerTypeCB.Text == "Convolution")
            {
                type = "c";
                if (result > 10)
                {
                    MessageBox.Show("Convolution's layer count is squared, must still be between 0 and 100");
                    return;
                }
            }
            LayerTypes.Add(type);
            LayerCounts.Add(result);
            LayerLB.Items.Add("[" + (LayerCounts.Count - 1).ToString() + "] " + LayerTypeCB.Text + ", " + result.ToString());
        }

        private void DelBtn_Click(object sender, EventArgs e)
        {
            if (LayerLB.Items.Count == 0) { return; }
            if (LayerLB.SelectedIndex == LayerTypes.Count - 1) { MessageBox.Show("Can't remove the output layer"); return; }
            LayerTypes.RemoveAt(LayerLB.SelectedIndex);
            LayerCounts.RemoveAt(LayerLB.SelectedIndex);
            LayerLB.Items.RemoveAt(LayerLB.SelectedIndex);
        }

        private void LayerCountTxt_TextChanged(object sender, EventArgs e)
        {
            if (LayerLB.SelectedIndex == LayerLB.Items.Count - 1) { return; }
            if (!(int.TryParse(LayerCountTxt.Text, out int result)) || result > 100 || result < 1)
            { LayerCountTxt.Text = 30.ToString(); MessageBox.Show("Layer count must be an int between 0 and 100\nReset to default"); return; }
        }

        private void UpdateBtn_Click(object sender, EventArgs e)
        {
            if (LayerLB.Items.Count == 0) { return; }
            if (LayerLB.SelectedIndex == LayerTypes.Count - 1) { MessageBox.Show("Can't change the output layer"); return; }
            int result = int.Parse(LayerCountTxt.Text);
            string type = null;
            if (LayerTypeCB.Text == "Fully Connected") { type = "f"; }
            if (LayerTypeCB.Text == "Convolution")
            {
                type = "c";
                if (result > 10)
                {
                    MessageBox.Show("Convolution's layer count is squared, must be between 0 and 10");
                    return;
                }
            }
            LayerTypes[LayerLB.SelectedIndex] = type;
            LayerCounts[LayerLB.SelectedIndex] = result;
            LayerLB.Items[LayerLB.SelectedIndex] = "[" + LayerLB.SelectedIndex.ToString() + "] " + LayerTypeCB.Text + ", " + result.ToString();

        }

        private void DefaultBtn_Click(object sender, EventArgs e)
        {
            LayerTypes = DefaultTypes();
            LayerCounts = DefaultCounts();
            NN newnn = ResetNN();
            RefreshListBoxes(newnn, true);
        }
        private NN ResetNN()
        {
            NN nn = new NN();
            if (LayerTypes is null || LayerTypes.Count == 0)
            {
                LayerTypes = DefaultTypes();
                LayerCounts = DefaultCounts();
            }
            nn.Init(GenerateLayers(COG.Checked), COG.Checked);
            IO.Write(nn, COG.Checked);
            return nn;
        }

        private void COG_CheckedChanged(object sender, EventArgs e)
        {
            var templcs = InactiveLayerCounts;
            InactiveLayerCounts = LayerCounts;
            LayerCounts = templcs;
            var templts = InactiveLayerTypes;
            InactiveLayerTypes = LayerTypes;
            LayerTypes = templts;

            if (LayerTypes is null || LayerTypes.Count == 0)
            {
                DefaultBtn_Click(sender, e); return;
            }
            NN newnn = ResetNN();
            RefreshListBoxes(newnn, true);
        }
    }
}
