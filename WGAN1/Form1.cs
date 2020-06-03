using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading;
using System.Windows.Forms;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Linq;
using System.Xml.Xsl;
using System.Drawing.Printing;

namespace WGAN1
{
    public partial class Form1 : Form
    {
        public List<string> ActiveLayers { get; set; }
        public List<string> InactiveLayers { get; set; }
        NN Critic;
        NN Generator;
        int batchsize = 10;
        int ctogratio = 5;
        int imgspeed = 0;

        int resolution = 28;
        int latentsize = 784;
        //The maximum RMSE allowed before the network stops learning
        public static int Cutoff = 10;
        bool dt;
        public bool DoneTraining { get { return dt; } set { dt = value; if (dt) { TrainBtn.Enabled = true; dt = false; } } }
        string cs;
        public string CScore { get { return cs; } set { cs = value; CScoreTxt.Text = value; } }
        string gs;
        public string GScore { get { return gs; } set { gs = value; GScoreTxt.Text = value; } }
        int e;
        public int Epoch { get { return e; } set { e = value; EpochTxt.Text = value.ToString(); } }
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
            LayerTypeCB.Items.Add("Sum");
            LayerTypeCB.Items.Add("Pool");
            LayerTypeCB.SelectedIndex = 0;

            Epoch = 0;
            InputNormCB.Checked = true;
            GradientNormCB.Checked = true;
            ClipTxt.Text = NN.ClipParameter.ToString();
            AlphaTxt.Text = NN.LearningRate.ToString();
            RMSDTxt.Text = NN.RMSDecay.ToString();
            MTxt.Text = batchsize.ToString();
            CTGTxt.Text = ctogratio.ToString();
            try
            {
                Critic = IO.Read(true);
                RefreshList(Critic, false);
                Generator = IO.Read(false);
                RefreshList(Generator, true);
            }
            catch
            {
                ActiveLayers = Default(false);
                InactiveLayers = Default(true);
                ResetBtn_Click(this, new EventArgs());
            }
            RefreshLayerLB();
            //Only want this shown if a conv layer is selected
            UpDownCB.Hide();
        }
        private void TrainBtn_Click(object sender, EventArgs e)
        {
            if (NN.Training) { NN.Training = false; TrainBtn.Enabled = false; return; }
            NN.Training = true;
            var thread = new Thread(() => 
            {
                 NN.Train(Critic, Generator, latentsize, resolution, batchsize, ctogratio, 1, this, imgspeed, InputNormCB.Checked, GradientNormCB.Checked);               
            });
            thread.IsBackground = true;
            thread.Start();
        }
        private void ResetBtn_Click(object sender, EventArgs e)
        {
            if (NN.Training) { NN.Save = false; NN.Training = false; TrainBtn.Enabled = false; }

            ResetNNs(COG.Checked);

            Epoch = 0;
        }
        List<bool> GenerateBatchnorms(bool cog)
        {
            List<string[]> layers = Split(ActiveLayers);
            if (COG.Checked != cog) { layers = Split(InactiveLayers); }
            List<bool> batchnorms = new List<bool>();
            foreach(string[] s in layers)
            {
                if (s.Length > 3 && s[3] == "1") { batchnorms.Add(true); continue; }
                batchnorms.Add(false); 
            }
            return batchnorms;
        }
        List<bool> GenerateResiduals(bool cog)
        {
            List<string[]> layers = Split(ActiveLayers);
            if (COG.Checked != cog) { layers = Split(InactiveLayers); }
            List<bool> residuals = new List<bool>();
            foreach(string[] s in layers)
            {
                if (s.Length > 2 && s[2] == "1") { residuals.Add(true); continue; }
                residuals.Add(false); 
            }
            return residuals;
        }
        List<bool> GenerateTanhs(bool cog)
        {
            List<string[]> layers = Split(ActiveLayers);
            if (COG.Checked != cog) { layers = Split(InactiveLayers); }
            List<bool> tanhs = new List<bool>();
            foreach (string[] s in layers)
            {
                if (s.Length > 4 && s[4] == "1") { tanhs.Add(true); continue; }
                tanhs.Add(false);
            }
            return tanhs;
        }
        List<iLayer> GenerateLayers(bool cog)
        {
            int priorsize;
            if (cog) { priorsize = resolution * resolution; }
            else { priorsize = latentsize; }
            List<string[]> layers = Split(ActiveLayers);
            if (COG.Checked != cog)
            {
                layers = Split(InactiveLayers);
            }
            List<iLayer> nnlayers = new List<iLayer>();
            for (int i = 0; i < layers.Count; i++)
            {
                int ncount;
                if (layers[i][0][0] == 's')
                {
                    nnlayers.Add(new SumLayer(priorsize, priorsize));
                    continue;
                }
                else
                {
                    ncount = int.Parse(layers[i][1]);
                }
                //If a convolution
                if (layers[i][0][0] == 'c')
                {
                    nnlayers.Add(new ConvolutionLayer(ncount, priorsize));
                    var convlayer = (nnlayers[i] as ConvolutionLayer);
                    int sqrtpriorsize = (int)Math.Sqrt(priorsize);

                    //Upscale if specified
                    if (layers[i][0].Length > 1)
                    {
                        convlayer.DownOrUp = false;
                        priorsize = sqrtpriorsize - 1;
                        if (layers[i].Length == 7)
                        {
                            convlayer.Stride = int.Parse(layers[i][6].ToString());
                            priorsize *= convlayer.Stride;

                            priorsize += ncount;

                            //Probably don't want to pad an upscaler, but w/e
                            if (int.TryParse(layers[i][5], out int result)) { convlayer.PadSize = result; }
                            //Default to 0 padding
                            else { convlayer.PadSize = 0; }

                            priorsize += 2 * convlayer.PadSize;
                        }
                        else
                        {
                            convlayer.PadSize = 0;
                            convlayer.Stride = 1;
                            priorsize += ncount;
                        }
                        nnlayers[i].OutputLength = priorsize;
                    }
                    //Otherwise downscale
                    else
                    {
                        convlayer.DownOrUp = true;
                        priorsize = (sqrtpriorsize - ncount);

                        if (layers[i].Length == 7)
                        {
                            convlayer.Stride = int.Parse(layers[i][6].ToString());
                            priorsize /= convlayer.Stride;
                            priorsize += 1;

                            if (int.TryParse(layers[i][5], out int result)) { convlayer.PadSize = result; }
                            //Calc pad size needed to return to the original size
                            else { convlayer.PadSize = (sqrtpriorsize - priorsize) / 2; }

                            priorsize += 2 * convlayer.PadSize;
                        }
                        else
                        {
                            convlayer.PadSize = 0;
                            convlayer.Stride = 1;
                            priorsize += 1;
                        }
                    }
                    priorsize *= priorsize;
                    nnlayers[i].OutputLength = priorsize;
                    continue;
                }
                if (layers[i][0][0] == 'f')
                {
                    nnlayers.Add(new FullyConnectedLayer(ncount, priorsize));
                    priorsize = ncount;
                    nnlayers[i].OutputLength = priorsize;
                    continue;
                }
                if (layers[i][0][0] == 'p')
                {
                    bool downorup = layers[i][0].Length < 2;
                    nnlayers.Add(new PoolingLayer(downorup, ncount, priorsize));
                    priorsize = nnlayers[i].Length;
                    nnlayers[i].OutputLength = priorsize;
                    continue;
                }
                throw new Exception("Invalid layer type");
            }
            return nnlayers;
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
            NN.Clear = true; CScore = null; GScore = null;
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
        private List<string> Default(bool cog)
        {
            //[0] is type of layer 
            //Convlayers downscale by default, add any char to make it upscale
            //[1] is count of layer
            //Sumlayers don't need a count, convlayers' count is kernelsize not outputsize
            //Defaulted to [false], [false], [false]
            //[2] is whether it is a residual
            //[3] is whether it batchnorms
            //[4] is whether to use Tanh
            //Defaulted to [0], [1]
            //[5] is padsize
            //[6] is stride
            var list = new List<string>();
            if (cog)
            {
                list.Add("c,5,0,0,1,0,1");
                list.Add("c,5,0,0,1,0,1");
                list.Add("c,5,1,1,1,0,1");

                list.Add("c,3,0,0,1,x,1");
                list.Add("c,3,0,0,1,x,1");
                list.Add("c,3,0,1,1,x,1");

                list.Add("s");
                list.Add("f,150,1,0,1");
                list.Add("f,100,1,0,1");
                list.Add("f,1,0,1,0");
            }
            else
            {
                //Residual
                list.Add("c,3,1,1,1,x,1");

                //Residue layer 1

                //2 padded conv layers + sum layer
                list.Add("c,7,0,1,1,x,1");
                list.Add("c,7,0,1,1,x,1");
                list.Add("s,-1,1");
                //Upscale
                list.Add("pu,2,0,1,1");
                //ConvT (residual)
                list.Add("cu,7,1,1,1,0,1");

                //Residue layer 2

                //2 padded conv layers + sum layer
                list.Add("c,7,0,1,1,x,1");
                list.Add("c,7,0,1,1,x,1");
                list.Add("s,-1,1");
                //Upscale
                list.Add("pu,2,0,1,1");
                //ConvT (residual)
                list.Add("cu,7,1,1,1,0,1");

                //Residue layer 3

                //2 padded conv layers + sum layer
                list.Add("c,7,0,1,1,x,1");
                list.Add("c,7,0,1,1,x,1");
                list.Add("s,-1,1");
                //Upscale
                list.Add("pu,2,0,1,1");
                //ConvT (residual)
                list.Add("cu,7,1,1,1,0,1");

                list.Add("c,17,0,1,1,0,3");
                list.Add("c,11,0,1,1,0,2");
                list.Add("c,10,0,1,1,0,1");

                //list.Add("s, -1, 1");
            }
            return list;
        }
        private void RefreshList(NN desired, bool ActiveOrInactive)
        {
            var layers = new List<string>();
            if (ActiveOrInactive) { LayerLB.Items.Clear(); }
            int index = 0;
            for (int i = 0; i < desired.Layers.Count; i++)
            {
                string layer = "";

                //Filter by layer type
                if (desired.Layers[i] is FullyConnectedLayer) 
                { 
                    layer += "f,"; layer += desired.Layers[i].Length + ",";
                    if (!(desired.ResidualLayers is null))
                    {
                        layer += desired.ResidualLayers[i] ? "1," : "0,";
                        layer += desired.BatchNormLayers[i] ? "1," : "0,";
                        layer += desired.TanhLayers[i] ? "1," : "0,";
                    }
                }
                if (desired.Layers[i] is ConvolutionLayer) 
                {
                    var conv = desired.Layers[i] as ConvolutionLayer;
                    layer += "c";
                    layer += conv.DownOrUp ? "," : "u,";
                    layer += conv.KernelSize + ",";
                    layer += desired.ResidualLayers[i] ? "1," : "0,";
                    layer += desired.BatchNormLayers[i] ? "1," : "0,";
                    layer += desired.TanhLayers[i] ? "1," : "0,";
                    layer += conv.PadSize.ToString() + ",";
                    layer += conv.Stride.ToString();
                }
                if (desired.Layers[i] is SumLayer) 
                {
                    layer += "s";
                }
                if (desired.Layers[i] is PoolingLayer)
                {
                    var pool = desired.Layers[i] as PoolingLayer;
                    layer += "p"; layer += pool.DownOrUp ? "," : "u,";
                    layer += pool.PoolSize + ",";
                    if (!(desired.ResidualLayers is null))
                    {
                        layer += desired.ResidualLayers[i] ? "1," : "0,";
                        layer += desired.BatchNormLayers[i] ? "1," : "0,";
                        layer += desired.TanhLayers[i] ? "1," : "0,";
                    }
                }
                layers.Add(layer);
                index++;
            }
            if (ActiveOrInactive)
            {
                ActiveLayers = layers;
                RefreshLayerLB();
            }
            else
            {
                InactiveLayers = layers;
            }
        }
        private void LayerLB_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (LayerLB.SelectedIndex < 0) { return; }
            var layers = Split(ActiveLayers);
            if (ActiveLayers[LayerLB.SelectedIndex][0] == 's')
            {
                LayerCountTxt.Text = null;
            }
            else
            {
                LayerCountTxt.Text = layers[LayerLB.SelectedIndex][1].ToString();
            }
            if (layers[LayerLB.SelectedIndex].Length > 2) { ResidualCB.Checked = layers[LayerLB.SelectedIndex][2] == "1"; }
            else { ResidualCB.Checked = false; }
            if (layers[LayerLB.SelectedIndex].Length > 3) { BatchnormCB.Checked = layers[LayerLB.SelectedIndex][3] == "1"; }
            else { BatchnormCB.Checked = false; }
            if (layers[LayerLB.SelectedIndex].Length > 4) { TanhCB.Checked = layers[LayerLB.SelectedIndex][4] == "1"; }
            else { TanhCB.Checked = false; }

            if (ActiveLayers[LayerLB.SelectedIndex][0] == 'c')
            { UpDownCB.Show(); UpDownCB.Checked = ActiveLayers[LayerLB.SelectedIndex][1] != ','; }
            else { UpDownCB.Hide(); }
        }

        private void UpBtn_Click(object sender, EventArgs e)
        {
            if (LayerLB.Items.Count < 2) { return; }
            if (LayerLB.SelectedIndex == 0) { return; }

            Swap(true, LayerLB.SelectedIndex, LayerLB.SelectedIndex - 1);

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

            Swap(true, LayerLB.SelectedIndex, LayerLB.SelectedIndex + 1);

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
            bool valid = int.TryParse(LayerCountTxt.Text, out int result);
            if (!valid && LayerTypeCB.Text != "Sum") { return; }
            string type = null;
            if (LayerTypeCB.Text == "Fully Connected") { type = "f"; }
            if (LayerTypeCB.Text == "Convolution")
            {
                type = "c";
                type.Append(UpDownCB.Checked ? 'u' : 'd');
                if (result > 10)
                {
                    MessageBox.Show("Convolution's layer count is squared, must still be between 0 and 100");
                    return;
                }
            }
            if (LayerTypeCB.Text == "Sum") { type = "s"; }
            if (LayerTypeCB.Text == "Pool") { type = "p"; }

            string residual = null, batchnorm = null, tanh = null;
            if (!(type == "s" || type == "p"))
            {
                residual = ResidualCB.Checked ? "1" : "0";
                batchnorm = BatchnormCB.Checked ? "1" : "0";
                tanh = TanhCB.Checked ? "1" : "0";
            }
            if (residual is null && batchnorm is null && tanh is null) 
            {
                if (!valid) { ActiveLayers.Add(type); }
                else { ActiveLayers.Add(type + "," + result.ToString()); }
            }
            else { ActiveLayers.Add(type + "," + result.ToString() + "," + residual + "," + batchnorm + "," + tanh); }
            if (!valid) { LayerLB.Items.Add("[" + (ActiveLayers.Count - 1).ToString() + "] " + LayerTypeCB.Text); }
            else { LayerLB.Items.Add("[" + (ActiveLayers.Count - 1).ToString() + "] " + LayerTypeCB.Text + ", " + result.ToString()); }
          
        }

        private void DelBtn_Click(object sender, EventArgs e)
        {
            if (LayerLB.Items.Count == 1) { MessageBox.Show("Can't remove the last item"); return; }
            if (LayerLB.SelectedIndex == ActiveLayers.Count - 1) { MessageBox.Show("Can't remove the output layer"); return; }
            ActiveLayers.RemoveAt(LayerLB.SelectedIndex);
            LayerLB.Items.RemoveAt(LayerLB.SelectedIndex);
        }

        private void LayerCountTxt_TextChanged(object sender, EventArgs e)
        {
            if (LayerLB.SelectedIndex == -1) { return; }
            if (ActiveLayers[LayerLB.SelectedIndex].ToString()[0] == 's') { return; }
            if (LayerLB.SelectedIndex == LayerLB.Items.Count - 1) { return; }

            if (!(int.TryParse(LayerCountTxt.Text, out int result)) || result > 100 || result < 1)
            {
                if (LayerLB.Items[LayerLB.SelectedIndex].ToString()[4] == 'C')
                {
                    LayerCountTxt.Text = 30.ToString();
                }
                else
                {
                    LayerCountTxt.Text = 5.ToString();
                }
                MessageBox.Show("Layer count must be an int between 0 and 100\nReset to default"); return; 
            }
            if (result % 2 == 0 && LayerLB.Items[LayerLB.SelectedIndex].ToString()[4] == 'C')
            {
                LayerCountTxt.Text = (result - 1).ToString(); MessageBox.Show("Convolution layers must have odd kernel sizes to allow for padding");
            }
        }

        private void UpdateBtn_Click(object sender, EventArgs e)
        {
            if (LayerLB.Items.Count == 0) { return; }
            if (LayerLB.SelectedIndex == ActiveLayers.Count - 1) { MessageBox.Show("Can't change the output layer"); return; }
            bool valid = int.TryParse(LayerCountTxt.Text, out int result);
            if (!valid) { return; }
            string type = null;
            if (LayerTypeCB.Text == "Fully Connected") { type = "f"; }
            if (LayerTypeCB.Text == "Convolution")
            {
                type = "c";
                type.Append(UpDownCB.Checked ? 'u' : 'd');
                if (result > 10)
                {
                    MessageBox.Show("Convolution's layer count is squared, must be between 0 and 10");
                    return;
                }
            }
            if (LayerTypeCB.Text == "Sum") { type = "s"; }

            string residual = ResidualCB.Checked ? "1" : "0";
            string batchnorm = BatchnormCB.Checked ? "1" : "0";
            string tanh = TanhCB.Checked ? "1" : "0";

            ActiveLayers[LayerLB.SelectedIndex] = type + result.ToString() + residual + batchnorm + tanh;
            LayerLB.Items[LayerLB.SelectedIndex] = "[" + LayerLB.SelectedIndex.ToString() + "] " + LayerTypeCB.Text + ", " + result.ToString();
        }

        private void DefaultBtn_Click(object sender, EventArgs e)
        {
            ActiveLayers = Default(COG.Checked);
            InactiveLayers = Default(!COG.Checked);
            RefreshLayerLB();
        }
        private void ResetNNs(bool cog)
        {
            if (ActiveLayers is null || ActiveLayers.Count == 0)
            {
                ActiveLayers = Default(cog);
            }
            if (InactiveLayers is null || InactiveLayers.Count == 0)
            {
                InactiveLayers = Default(!cog);
            }
            Generator = new NN().Init(GenerateLayers(false), GenerateTanhs(false), GenerateResiduals(false), GenerateBatchnorms(false));
            Critic = new NN().Init(GenerateLayers(true), GenerateTanhs(true), GenerateResiduals(true), GenerateBatchnorms(true));
            IO.Write(Critic, true);
            IO.Write(Generator, false);
            if (cog) { OutputCountTxt.Text = Critic.OutputLength.ToString(); }
            else { OutputCountTxt.Text = Generator.OutputLength.ToString(); }
        }
        private void RefreshLayerLB()
        {
            LayerLB.Items.Clear();
            List<string[]> layers = Split(ActiveLayers);
            for (int i = 0; i < ActiveLayers.Count; i++)
            {
                string description = "";
                if (layers[i][0][0] == 'c') { description = "Convolution, "; }
                if (layers[i][0][0] == 'f') { description = "Fully Connected, "; }
                if (layers[i][0][0] == 's') { description = "Sum"; }
                if (layers[i][0][0] == 'p') { description = "Pool, "; }
                if (layers[i].Length > 1 && layers[i][0][0] != 's') { description += layers[i][1].ToString(); }
                LayerLB.Items.Add("[" + i + "] " + description);
            }
        }

        private void COG_CheckedChanged(object sender, EventArgs e)
        {
            var temp = InactiveLayers;
            InactiveLayers = ActiveLayers;
            ActiveLayers = temp;

            if (ActiveLayers is null || ActiveLayers.Count == 0
                || InactiveLayers is null || InactiveLayers.Count == 0)
            {
                DefaultBtn_Click(sender, e); return;
            }
            RefreshLayerLB();
            UpDownCB.Hide();
            LayerCountTxt.Text = null;
        }
        private void Swap(bool active, int indexA, int indexB)
        {
            if (active) { ActiveLayers = Swap(ActiveLayers, indexA, indexB); }
            else { InactiveLayers = Swap(InactiveLayers, indexA, indexB); }
        }
        public static List<T> Swap<T>(List<T> list, int indexA, int indexB)
        {
            var temp = list[indexA];
            list[indexA] = list[indexB];
            list[indexB] = temp;
            return list;
        }
        public List<string[]> Split(List<string> original)
        {
            List<string[]> output = new List<string[]>();
            foreach (string s in original)
            {
                output.Add(s.Split(','));
            }
            return output;
        }
    }
}
