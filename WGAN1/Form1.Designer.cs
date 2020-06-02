namespace WGAN1
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.NumPB = new System.Windows.Forms.PictureBox();
            this.AlphaTxt = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.RMSDTxt = new System.Windows.Forms.TextBox();
            this.MTxt = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.CTGTxt = new System.Windows.Forms.TextBox();
            this.TrainBtn = new System.Windows.Forms.Button();
            this.label5 = new System.Windows.Forms.Label();
            this.ClipTxt = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.CScoreTxt = new System.Windows.Forms.TextBox();
            this.ClearBtn = new System.Windows.Forms.Button();
            this.ResetBtn = new System.Windows.Forms.Button();
            this.LayerLB = new System.Windows.Forms.ListBox();
            this.UpBtn = new System.Windows.Forms.Button();
            this.DownBtn = new System.Windows.Forms.Button();
            this.DefaultBtn = new System.Windows.Forms.Button();
            this.UpdateBtn = new System.Windows.Forms.Button();
            this.LayerCountTxt = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.DelBtn = new System.Windows.Forms.Button();
            this.AddBtn = new System.Windows.Forms.Button();
            this.LayerTypeCB = new System.Windows.Forms.ComboBox();
            this.COG = new System.Windows.Forms.CheckBox();
            this.label8 = new System.Windows.Forms.Label();
            this.GScoreTxt = new System.Windows.Forms.TextBox();
            this.UpDownCB = new System.Windows.Forms.CheckBox();
            this.InputNormCB = new System.Windows.Forms.CheckBox();
            this.label9 = new System.Windows.Forms.Label();
            this.EpochTxt = new System.Windows.Forms.TextBox();
            this.GradientNormCB = new System.Windows.Forms.CheckBox();
            this.label10 = new System.Windows.Forms.Label();
            this.SkipTxt = new System.Windows.Forms.TextBox();
            this.ResidualCB = new System.Windows.Forms.CheckBox();
            this.BatchnormCB = new System.Windows.Forms.CheckBox();
            this.TanhCB = new System.Windows.Forms.CheckBox();
            ((System.ComponentModel.ISupportInitialize)(this.NumPB)).BeginInit();
            this.SuspendLayout();
            // 
            // NumPB
            // 
            this.NumPB.Location = new System.Drawing.Point(484, 43);
            this.NumPB.Name = "NumPB";
            this.NumPB.Size = new System.Drawing.Size(644, 611);
            this.NumPB.TabIndex = 0;
            this.NumPB.TabStop = false;
            // 
            // AlphaTxt
            // 
            this.AlphaTxt.Location = new System.Drawing.Point(109, 181);
            this.AlphaTxt.Name = "AlphaTxt";
            this.AlphaTxt.Size = new System.Drawing.Size(142, 31);
            this.AlphaTxt.TabIndex = 1;
            this.AlphaTxt.TextChanged += new System.EventHandler(this.AlphaTxt_TextChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(105, 153);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(147, 25);
            this.label1.TabIndex = 2;
            this.label1.Text = "Learning Rate";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(279, 153);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(120, 25);
            this.label2.TabIndex = 3;
            this.label2.Text = "RMSDecay";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(105, 215);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(115, 25);
            this.label3.TabIndex = 4;
            this.label3.Text = "Batch Size";
            // 
            // RMSDTxt
            // 
            this.RMSDTxt.Location = new System.Drawing.Point(257, 181);
            this.RMSDTxt.Name = "RMSDTxt";
            this.RMSDTxt.Size = new System.Drawing.Size(142, 31);
            this.RMSDTxt.TabIndex = 5;
            this.RMSDTxt.TextChanged += new System.EventHandler(this.RMSDTxt_TextChanged);
            // 
            // MTxt
            // 
            this.MTxt.Location = new System.Drawing.Point(109, 243);
            this.MTxt.Name = "MTxt";
            this.MTxt.Size = new System.Drawing.Size(142, 31);
            this.MTxt.TabIndex = 6;
            this.MTxt.TextChanged += new System.EventHandler(this.MTxt_TextChanged);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(10, 277);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(243, 25);
            this.label4.TabIndex = 8;
            this.label4.Text = "Critic to Generator Ratio";
            // 
            // CTGTxt
            // 
            this.CTGTxt.Location = new System.Drawing.Point(110, 305);
            this.CTGTxt.Name = "CTGTxt";
            this.CTGTxt.Size = new System.Drawing.Size(142, 31);
            this.CTGTxt.TabIndex = 7;
            this.CTGTxt.TextChanged += new System.EventHandler(this.CTGTxt_TextChanged);
            // 
            // TrainBtn
            // 
            this.TrainBtn.Location = new System.Drawing.Point(109, 389);
            this.TrainBtn.Name = "TrainBtn";
            this.TrainBtn.Size = new System.Drawing.Size(142, 45);
            this.TrainBtn.TabIndex = 9;
            this.TrainBtn.Text = "Train";
            this.TrainBtn.UseVisualStyleBackColor = true;
            this.TrainBtn.Click += new System.EventHandler(this.TrainBtn_Click);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(256, 277);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(195, 25);
            this.label5.TabIndex = 11;
            this.label5.Text = "Clipping Parameter";
            // 
            // ClipTxt
            // 
            this.ClipTxt.Location = new System.Drawing.Point(257, 305);
            this.ClipTxt.Name = "ClipTxt";
            this.ClipTxt.Size = new System.Drawing.Size(142, 31);
            this.ClipTxt.TabIndex = 10;
            this.ClipTxt.TextChanged += new System.EventHandler(this.ClipTxt_TextChanged);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(112, 449);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(128, 25);
            this.label6.TabIndex = 13;
            this.label6.Text = "Critic RMSE";
            // 
            // CScoreTxt
            // 
            this.CScoreTxt.Location = new System.Drawing.Point(117, 477);
            this.CScoreTxt.Name = "CScoreTxt";
            this.CScoreTxt.ReadOnly = true;
            this.CScoreTxt.Size = new System.Drawing.Size(142, 31);
            this.CScoreTxt.TabIndex = 12;
            // 
            // ClearBtn
            // 
            this.ClearBtn.Location = new System.Drawing.Point(257, 389);
            this.ClearBtn.Name = "ClearBtn";
            this.ClearBtn.Size = new System.Drawing.Size(142, 45);
            this.ClearBtn.TabIndex = 14;
            this.ClearBtn.Text = "Clear";
            this.ClearBtn.UseVisualStyleBackColor = true;
            this.ClearBtn.Click += new System.EventHandler(this.ClearBtn_Click);
            // 
            // ResetBtn
            // 
            this.ResetBtn.Location = new System.Drawing.Point(1495, 630);
            this.ResetBtn.Name = "ResetBtn";
            this.ResetBtn.Size = new System.Drawing.Size(142, 45);
            this.ResetBtn.TabIndex = 15;
            this.ResetBtn.Text = "Reset NN";
            this.ResetBtn.UseVisualStyleBackColor = true;
            this.ResetBtn.Click += new System.EventHandler(this.ResetBtn_Click);
            // 
            // LayerLB
            // 
            this.LayerLB.FormattingEnabled = true;
            this.LayerLB.ItemHeight = 25;
            this.LayerLB.Location = new System.Drawing.Point(1146, 43);
            this.LayerLB.Name = "LayerLB";
            this.LayerLB.Size = new System.Drawing.Size(404, 329);
            this.LayerLB.TabIndex = 16;
            this.LayerLB.SelectedIndexChanged += new System.EventHandler(this.LayerLB_SelectedIndexChanged);
            // 
            // UpBtn
            // 
            this.UpBtn.Location = new System.Drawing.Point(1146, 391);
            this.UpBtn.Name = "UpBtn";
            this.UpBtn.Size = new System.Drawing.Size(142, 45);
            this.UpBtn.TabIndex = 17;
            this.UpBtn.Text = "Up";
            this.UpBtn.UseVisualStyleBackColor = true;
            this.UpBtn.Click += new System.EventHandler(this.UpBtn_Click);
            // 
            // DownBtn
            // 
            this.DownBtn.Location = new System.Drawing.Point(1310, 391);
            this.DownBtn.Name = "DownBtn";
            this.DownBtn.Size = new System.Drawing.Size(142, 45);
            this.DownBtn.TabIndex = 18;
            this.DownBtn.Text = "Down";
            this.DownBtn.UseVisualStyleBackColor = true;
            this.DownBtn.Click += new System.EventHandler(this.DownBtn_Click);
            // 
            // DefaultBtn
            // 
            this.DefaultBtn.Location = new System.Drawing.Point(1146, 442);
            this.DefaultBtn.Name = "DefaultBtn";
            this.DefaultBtn.Size = new System.Drawing.Size(142, 45);
            this.DefaultBtn.TabIndex = 19;
            this.DefaultBtn.Text = "Default";
            this.DefaultBtn.UseVisualStyleBackColor = true;
            this.DefaultBtn.Click += new System.EventHandler(this.DefaultBtn_Click);
            // 
            // UpdateBtn
            // 
            this.UpdateBtn.Location = new System.Drawing.Point(1310, 442);
            this.UpdateBtn.Name = "UpdateBtn";
            this.UpdateBtn.Size = new System.Drawing.Size(142, 45);
            this.UpdateBtn.TabIndex = 20;
            this.UpdateBtn.Text = "Update";
            this.UpdateBtn.UseVisualStyleBackColor = true;
            this.UpdateBtn.Click += new System.EventHandler(this.UpdateBtn_Click);
            // 
            // LayerCountTxt
            // 
            this.LayerCountTxt.Location = new System.Drawing.Point(1146, 570);
            this.LayerCountTxt.Name = "LayerCountTxt";
            this.LayerCountTxt.Size = new System.Drawing.Size(142, 31);
            this.LayerCountTxt.TabIndex = 22;
            this.LayerCountTxt.TextChanged += new System.EventHandler(this.LayerCountTxt_TextChanged);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(1146, 542);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(129, 25);
            this.label7.TabIndex = 21;
            this.label7.Text = "Layer Count";
            // 
            // DelBtn
            // 
            this.DelBtn.Location = new System.Drawing.Point(1310, 493);
            this.DelBtn.Name = "DelBtn";
            this.DelBtn.Size = new System.Drawing.Size(142, 45);
            this.DelBtn.TabIndex = 24;
            this.DelBtn.Text = "Del";
            this.DelBtn.UseVisualStyleBackColor = true;
            this.DelBtn.Click += new System.EventHandler(this.DelBtn_Click);
            // 
            // AddBtn
            // 
            this.AddBtn.Location = new System.Drawing.Point(1146, 493);
            this.AddBtn.Name = "AddBtn";
            this.AddBtn.Size = new System.Drawing.Size(142, 45);
            this.AddBtn.TabIndex = 23;
            this.AddBtn.Text = "Add";
            this.AddBtn.UseVisualStyleBackColor = true;
            this.AddBtn.Click += new System.EventHandler(this.AddBtn_Click);
            // 
            // LayerTypeCB
            // 
            this.LayerTypeCB.FormattingEnabled = true;
            this.LayerTypeCB.Location = new System.Drawing.Point(1294, 568);
            this.LayerTypeCB.Name = "LayerTypeCB";
            this.LayerTypeCB.Size = new System.Drawing.Size(154, 33);
            this.LayerTypeCB.TabIndex = 25;
            this.LayerTypeCB.Text = "Layer Type";
            // 
            // COG
            // 
            this.COG.AutoSize = true;
            this.COG.Location = new System.Drawing.Point(1168, 611);
            this.COG.Name = "COG";
            this.COG.Size = new System.Drawing.Size(280, 29);
            this.COG.TabIndex = 26;
            this.COG.Text = "Critic [1] or Generator [0]";
            this.COG.UseVisualStyleBackColor = true;
            this.COG.CheckedChanged += new System.EventHandler(this.COG_CheckedChanged);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(261, 449);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(175, 25);
            this.label8.TabIndex = 28;
            this.label8.Text = "Generator RMSE";
            // 
            // GScoreTxt
            // 
            this.GScoreTxt.Location = new System.Drawing.Point(266, 477);
            this.GScoreTxt.Name = "GScoreTxt";
            this.GScoreTxt.ReadOnly = true;
            this.GScoreTxt.Size = new System.Drawing.Size(142, 31);
            this.GScoreTxt.TabIndex = 27;
            // 
            // UpDownCB
            // 
            this.UpDownCB.AutoSize = true;
            this.UpDownCB.Location = new System.Drawing.Point(1168, 646);
            this.UpDownCB.Name = "UpDownCB";
            this.UpDownCB.Size = new System.Drawing.Size(167, 29);
            this.UpDownCB.TabIndex = 29;
            this.UpDownCB.Text = "Downsample";
            this.UpDownCB.UseVisualStyleBackColor = true;
            // 
            // InputNormCB
            // 
            this.InputNormCB.AutoSize = true;
            this.InputNormCB.Location = new System.Drawing.Point(257, 342);
            this.InputNormCB.Name = "InputNormCB";
            this.InputNormCB.Size = new System.Drawing.Size(204, 29);
            this.InputNormCB.TabIndex = 30;
            this.InputNormCB.Text = "Normalize Inputs";
            this.InputNormCB.UseVisualStyleBackColor = true;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(197, 511);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(150, 25);
            this.label9.TabIndex = 32;
            this.label9.Text = "Current Epoch";
            // 
            // EpochTxt
            // 
            this.EpochTxt.Location = new System.Drawing.Point(202, 539);
            this.EpochTxt.Name = "EpochTxt";
            this.EpochTxt.ReadOnly = true;
            this.EpochTxt.Size = new System.Drawing.Size(142, 31);
            this.EpochTxt.TabIndex = 31;
            // 
            // GradientNormCB
            // 
            this.GradientNormCB.AutoSize = true;
            this.GradientNormCB.Location = new System.Drawing.Point(57, 342);
            this.GradientNormCB.Name = "GradientNormCB";
            this.GradientNormCB.Size = new System.Drawing.Size(194, 29);
            this.GradientNormCB.TabIndex = 33;
            this.GradientNormCB.Text = "Norm Gradients";
            this.GradientNormCB.UseVisualStyleBackColor = true;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(272, 215);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(127, 25);
            this.label10.TabIndex = 35;
            this.label10.Text = "Skip Iterator";
            // 
            // SkipTxt
            // 
            this.SkipTxt.Location = new System.Drawing.Point(257, 243);
            this.SkipTxt.Name = "SkipTxt";
            this.SkipTxt.Size = new System.Drawing.Size(142, 31);
            this.SkipTxt.TabIndex = 34;
            // 
            // ResidualCB
            // 
            this.ResidualCB.AutoSize = true;
            this.ResidualCB.Location = new System.Drawing.Point(1168, 681);
            this.ResidualCB.Name = "ResidualCB";
            this.ResidualCB.Size = new System.Drawing.Size(214, 29);
            this.ResidualCB.TabIndex = 36;
            this.ResidualCB.Text = "Save for Residual";
            this.ResidualCB.UseVisualStyleBackColor = true;
            // 
            // BatchnormCB
            // 
            this.BatchnormCB.AutoSize = true;
            this.BatchnormCB.Location = new System.Drawing.Point(1168, 719);
            this.BatchnormCB.Name = "BatchnormCB";
            this.BatchnormCB.Size = new System.Drawing.Size(147, 29);
            this.BatchnormCB.TabIndex = 37;
            this.BatchnormCB.Text = "Batchnorm";
            this.BatchnormCB.UseVisualStyleBackColor = true;
            // 
            // ReLuCB
            // 
            this.TanhCB.AutoSize = true;
            this.TanhCB.Location = new System.Drawing.Point(1321, 719);
            this.TanhCB.Name = "ReLuCB";
            this.TanhCB.Size = new System.Drawing.Size(139, 29);
            this.TanhCB.TabIndex = 38;
            this.TanhCB.Text = "Use ReLu";
            this.TanhCB.UseVisualStyleBackColor = true;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(12F, 25F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1743, 760);
            this.Controls.Add(this.TanhCB);
            this.Controls.Add(this.BatchnormCB);
            this.Controls.Add(this.ResidualCB);
            this.Controls.Add(this.label10);
            this.Controls.Add(this.SkipTxt);
            this.Controls.Add(this.GradientNormCB);
            this.Controls.Add(this.label9);
            this.Controls.Add(this.EpochTxt);
            this.Controls.Add(this.InputNormCB);
            this.Controls.Add(this.UpDownCB);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.GScoreTxt);
            this.Controls.Add(this.COG);
            this.Controls.Add(this.LayerTypeCB);
            this.Controls.Add(this.DelBtn);
            this.Controls.Add(this.AddBtn);
            this.Controls.Add(this.LayerCountTxt);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.UpdateBtn);
            this.Controls.Add(this.DefaultBtn);
            this.Controls.Add(this.DownBtn);
            this.Controls.Add(this.UpBtn);
            this.Controls.Add(this.LayerLB);
            this.Controls.Add(this.ResetBtn);
            this.Controls.Add(this.ClearBtn);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.CScoreTxt);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.ClipTxt);
            this.Controls.Add(this.TrainBtn);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.CTGTxt);
            this.Controls.Add(this.MTxt);
            this.Controls.Add(this.RMSDTxt);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.AlphaTxt);
            this.Controls.Add(this.NumPB);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.NumPB)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox NumPB;
        private System.Windows.Forms.TextBox AlphaTxt;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox RMSDTxt;
        private System.Windows.Forms.TextBox MTxt;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox CTGTxt;
        private System.Windows.Forms.Button TrainBtn;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox ClipTxt;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox CScoreTxt;
        private System.Windows.Forms.Button ClearBtn;
        private System.Windows.Forms.Button ResetBtn;
        private System.Windows.Forms.ListBox LayerLB;
        private System.Windows.Forms.Button UpBtn;
        private System.Windows.Forms.Button DownBtn;
        private System.Windows.Forms.Button DefaultBtn;
        private System.Windows.Forms.Button UpdateBtn;
        private System.Windows.Forms.TextBox LayerCountTxt;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Button DelBtn;
        private System.Windows.Forms.Button AddBtn;
        private System.Windows.Forms.ComboBox LayerTypeCB;
        private System.Windows.Forms.CheckBox COG;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox GScoreTxt;
        private System.Windows.Forms.CheckBox UpDownCB;
        private System.Windows.Forms.CheckBox InputNormCB;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TextBox EpochTxt;
        private System.Windows.Forms.CheckBox GradientNormCB;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.TextBox SkipTxt;
        private System.Windows.Forms.CheckBox ResidualCB;
        private System.Windows.Forms.CheckBox BatchnormCB;
        private System.Windows.Forms.CheckBox TanhCB;
    }
}

