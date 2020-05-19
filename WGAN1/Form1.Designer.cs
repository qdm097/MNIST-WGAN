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
            ((System.ComponentModel.ISupportInitialize)(this.NumPB)).BeginInit();
            this.SuspendLayout();
            // 
            // NumPB
            // 
            this.NumPB.Location = new System.Drawing.Point(492, 107);
            this.NumPB.Name = "NumPB";
            this.NumPB.Size = new System.Drawing.Size(644, 611);
            this.NumPB.TabIndex = 0;
            this.NumPB.TabStop = false;
            // 
            // AlphaTxt
            // 
            this.AlphaTxt.Location = new System.Drawing.Point(166, 206);
            this.AlphaTxt.Name = "AlphaTxt";
            this.AlphaTxt.Size = new System.Drawing.Size(142, 31);
            this.AlphaTxt.TabIndex = 1;
            this.AlphaTxt.TextChanged += new System.EventHandler(this.AlphaTxt_TextChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(161, 178);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(147, 25);
            this.label1.TabIndex = 2;
            this.label1.Text = "Learning Rate";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(166, 240);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(120, 25);
            this.label2.TabIndex = 3;
            this.label2.Text = "RMSDecay";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(166, 302);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(115, 25);
            this.label3.TabIndex = 4;
            this.label3.Text = "Batch Size";
            // 
            // RMSDTxt
            // 
            this.RMSDTxt.Location = new System.Drawing.Point(166, 268);
            this.RMSDTxt.Name = "RMSDTxt";
            this.RMSDTxt.Size = new System.Drawing.Size(142, 31);
            this.RMSDTxt.TabIndex = 5;
            this.RMSDTxt.TextChanged += new System.EventHandler(this.RMSDTxt_TextChanged);
            // 
            // MTxt
            // 
            this.MTxt.Location = new System.Drawing.Point(166, 330);
            this.MTxt.Name = "MTxt";
            this.MTxt.Size = new System.Drawing.Size(142, 31);
            this.MTxt.TabIndex = 6;
            this.MTxt.TextChanged += new System.EventHandler(this.MTxt_TextChanged);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(166, 364);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(243, 25);
            this.label4.TabIndex = 8;
            this.label4.Text = "Critic to Generator Ratio";
            // 
            // CTGTxt
            // 
            this.CTGTxt.Location = new System.Drawing.Point(166, 392);
            this.CTGTxt.Name = "CTGTxt";
            this.CTGTxt.Size = new System.Drawing.Size(142, 31);
            this.CTGTxt.TabIndex = 7;
            this.CTGTxt.TextChanged += new System.EventHandler(this.CTGTxt_TextChanged);
            // 
            // TrainBtn
            // 
            this.TrainBtn.Location = new System.Drawing.Point(166, 492);
            this.TrainBtn.Name = "TrainBtn";
            this.TrainBtn.Size = new System.Drawing.Size(142, 35);
            this.TrainBtn.TabIndex = 9;
            this.TrainBtn.Text = "Train";
            this.TrainBtn.UseVisualStyleBackColor = true;
            this.TrainBtn.Click += new System.EventHandler(this.TrainBtn_Click);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(161, 427);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(195, 25);
            this.label5.TabIndex = 11;
            this.label5.Text = "Clipping Parameter";
            // 
            // ClipTxt
            // 
            this.ClipTxt.Location = new System.Drawing.Point(166, 455);
            this.ClipTxt.Name = "ClipTxt";
            this.ClipTxt.Size = new System.Drawing.Size(142, 31);
            this.ClipTxt.TabIndex = 10;
            this.ClipTxt.TextChanged += new System.EventHandler(this.ClipTxt_TextChanged);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(161, 575);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(123, 25);
            this.label6.TabIndex = 13;
            this.label6.Text = "Critic Score";
            // 
            // CScoreTxt
            // 
            this.CScoreTxt.Location = new System.Drawing.Point(166, 603);
            this.CScoreTxt.Name = "CScoreTxt";
            this.CScoreTxt.ReadOnly = true;
            this.CScoreTxt.Size = new System.Drawing.Size(142, 31);
            this.CScoreTxt.TabIndex = 12;
            // 
            // ClearBtn
            // 
            this.ClearBtn.Location = new System.Drawing.Point(166, 640);
            this.ClearBtn.Name = "ClearBtn";
            this.ClearBtn.Size = new System.Drawing.Size(142, 35);
            this.ClearBtn.TabIndex = 14;
            this.ClearBtn.Text = "Clear";
            this.ClearBtn.UseVisualStyleBackColor = true;
            this.ClearBtn.Click += new System.EventHandler(this.ClearBtn_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(12F, 25F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1189, 760);
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
    }
}

