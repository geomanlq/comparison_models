import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.StringTokenizer;


 
public class InitTensor {

	public int rank;            // 张量秩
	public int trainRound = 1000;    //训练轮数 
	public int minRMSERound = 0;     //获取最小RMSE的轮数 
	public int minMAERound = 0;      //获取最小MAE的轮数  
	public int delayRound = 50;      //获取最小轮数后继续测试轮数
	public double minRMSE = 100;   
	public double minMAE = 100;  
	public double everyRoundRMSE[], everyRoundMAE[]; // 定义两个数组，存储验证集每一轮的RMSE，和MAE
	public double everyRoundRMSE2[], everyRoundMAE2[]; // 定义两个数组，存储测试集每一轮的RMSE，和MAE

	public double con;
	public double errorgap ;      //早停误差间距
	
	public ArrayList<TensorTuple> trainData = new ArrayList<TensorTuple>();
	public ArrayList<TensorTuple> testData = new ArrayList<TensorTuple>();
	public ArrayList<TensorTuple> validData = new ArrayList<TensorTuple>();
	public int trainCount = 0;   //测试集数目
	public int testCount = 0;   //测试集数目
	public int validCount = 0;  //验证集数目
	public int maxAID = 0, maxBID = 0, maxCID = 0; // 获取张量三个维度的维度值 
	public int minAID = Integer.MAX_VALUE, minBID = Integer.MAX_VALUE, minCID = Integer.MAX_VALUE; // 获取张量三个维度起始值

	public double maxValue = -1; // 获取最大记录数值 
	public double minValue = Integer.MAX_VALUE; // 获取最小记录数值 
	
	public double[] aSliceSet, bSliceSet, cSliceSet; // 定义三个矩阵相关联的个
	
	public double[][] S, D, T; // 定义三个输出因子矩阵和训练因子矩阵
	public double[][] Sup, Sdown, Dup, Ddown, Tup, Tdown; // 辅助矩阵

	public double[] a, b, c; //bias输出参数向量和训练参数向量
	public double[] aup, adown, bup, bdown, cup, cdown; //bias训练参数向量辅助向量
	
	public String trainFile = null;
	public String testFile = null;
	public String separator = null;	
	public String validFile = null;
	
	protected InitTensor(String trainFile, String validFile, String testFile, String separator)
	{
		this.trainFile = trainFile;
		this.validFile = validFile;
		this.testFile = testFile;
		this.separator = separator; 

	}
	
	//T=0输入训练集，T=1输入测试集
	public void initData(String inputFile, ArrayList<TensorTuple> data, int T) throws IOException
	{
		File input = new File(inputFile);
		BufferedReader in = new BufferedReader(new FileReader(input));
		String inTemp;
		while((inTemp = in.readLine()) != null)
		{
			StringTokenizer st = new StringTokenizer(inTemp, separator);
			
			String iTemp = null; 
			if(st.hasMoreTokens())
				iTemp = st.nextToken();
			 
			String jTemp = null;
			if(st.hasMoreTokens())
				jTemp = st.nextToken();
			 
			String kTemp = null;
			if(st.hasMoreTokens())
				kTemp = st.nextToken();
			
			String tValueTemp = null;
			if(st.hasMoreTokens())
				tValueTemp = st.nextToken();
			

			
			int aid = Integer.valueOf(iTemp);
			int bid = Integer.valueOf(jTemp);
			int cid = Integer.valueOf(kTemp);
			double value = Double.valueOf(tValueTemp);


			TensorTuple qtemp = new TensorTuple();
			qtemp.aID = aid;
			qtemp.bID = bid;
			qtemp.cID = cid;
			qtemp.value = value;

			if (qtemp.cID > 2)
			{
				data.add(qtemp);
			}
				
			//获取训练数据张量每个模的大小
			this.maxAID = (this.maxAID > aid) ? this.maxAID : aid;
			this.maxBID = (this.maxBID > bid) ? this.maxBID : bid;
			this.maxCID = (this.maxCID > cid) ? this.maxCID : cid;


			this.minAID = (this.minAID < aid) ? this.minAID : aid;
			this.minBID = (this.minBID < bid) ? this.minBID : bid;
			this.minCID = (this.minCID < cid) ? this.minCID : cid;

			this.maxValue = (this.maxValue > value) ? this.maxValue : value;
			this.minValue = (this.minValue < value) ? this.minValue : value;

			this.maxAID = Math.max(this.maxAID, this.maxBID);
			this.maxBID = Math.max(this.maxAID, this.maxBID);

			if(T == 1)
			{
				this.trainCount++;	   //统计训练集数目

			}
			else if(T == 2)
			{
				this.validCount++;
			}else if(T == 3)
			{
				this.testCount++;
			}
		}
		
		in.close();
		
	//	System.out.println("testCount "+testCount);
	//	System.out.println("trainCount "+trainCount);
		

	}

	
	//一定范围随机初始化因子矩阵
	public int scale = 1000;
	public double initscale;  //不同数据集取值可能不同
	public double minvalue;
	public void initFactorMatrix() {
		S = new double[this.maxAID + 1][this.rank+1];
		D = new double[this.maxBID + 1][this.rank+1];
		T = new double[this.maxCID + 1][this.rank+1];
		
		a = new double[this.maxAID + 1];
		b = new double[this.maxBID + 1];
		c = new double[this.maxCID + 1];
		
		everyRoundRMSE = new double[this.trainRound+1];
		everyRoundMAE = new double[this.trainRound+1]; 
		
		everyRoundRMSE[0] = minRMSE; 
		everyRoundMAE[0] = minMAE; 
		
		everyRoundRMSE2 = new double[this.trainRound+1];
		everyRoundMAE2 = new double[this.trainRound+1]; 
		
		everyRoundRMSE2[0] = minRMSE;
		everyRoundMAE2[0] = minMAE;

		con = 0;

		Random random = new Random();
		for (int a_id = 1; a_id <= maxAID; a_id++) {
			for (int r = 1; r <= rank; r++) {
//				S[a_id][r] = initscale * random.nextInt(scale) / scale;
				S[a_id][r] = minvalue + random.nextDouble() * (initscale - minvalue);
			}
			
//			a[a_id] = initscale * random.nextInt(scale) / scale;
			a[a_id] = minvalue + random.nextDouble() * (initscale - minvalue);
		}
		
		for (int b_id = 1; b_id <= maxBID; b_id++) {
			for (int r = 1; r <= rank; r++) {
//				D[b_id][r] = initscale * random.nextInt(scale) / scale;
				D[b_id][r] = minvalue + random.nextDouble() * (initscale - minvalue);
			}
			
//			b[b_id] = initscale * random.nextInt(scale) / scale;
			b[b_id] = minvalue + random.nextDouble() * (initscale - minvalue);
		}
		
		for (int c_id = 1; c_id <= maxCID; c_id++) {
			for (int r = 1; r <= rank; r++) {
//				T[c_id][r] = initscale * random.nextInt(scale) / scale;
				T[c_id][r] = minvalue + random.nextDouble() * (initscale - minvalue);
			}
			
//			c[c_id] = initscale * random.nextInt(scale) / scale;
			c[c_id] = minvalue + random.nextDouble() * (initscale - minvalue);
		}
		
	}
 
	

	//记录在训练集张量中每个方向上切片矩阵中包含元素个数
	public void initSliceSet() {
		aSliceSet = new double[maxAID + 1];
		bSliceSet = new double[maxBID + 1];
		cSliceSet = new double[maxCID + 1];
		for (TensorTuple tensor_tuple : trainData) {
			aSliceSet[tensor_tuple.aID] += 1;
			bSliceSet[tensor_tuple.bID] += 1;
			cSliceSet[tensor_tuple.cID] += 1;
		}
		
		for (int a_id = 1; a_id <= maxAID; a_id++) 
		{
			if(aSliceSet[a_id] == 0)
				aSliceSet[a_id] = 1;
		}
		
		for (int b_id = 1; b_id <= maxBID; b_id++) 
		{
			if(bSliceSet[b_id] == 0)
				bSliceSet[b_id] = 1;
		}
		
		for (int c_id = 1; c_id <= maxCID; c_id++) 
		{
			if(cSliceSet[c_id] == 0)
				cSliceSet[c_id] = 1;
		}

	}


	// 初始化辅助矩阵
	public void initAssistMatrix() {
		
		Sup = new double[maxAID + 1][this.rank+1];
		Sdown = new double[maxAID + 1][this.rank+1];
		Dup = new double[maxBID + 1][this.rank+1];
		Ddown = new double[maxBID + 1][this.rank+1];
		Tup = new double[maxCID + 1][this.rank+1];
		Tdown = new double[maxCID + 1][this.rank+1];
		
		aup = new double[maxAID + 1];
		adown = new double[maxAID + 1];
		bup = new double[maxBID + 1];  
		bdown = new double[maxBID + 1];
		cup = new double[maxCID + 1];
		cdown = new double[maxCID + 1];

		
		for (int max_a_id = 1; max_a_id <= maxAID; max_a_id++) 
		{
			for (int r = 1; r <= rank; r++) 
			{
				Sup[max_a_id][r] = 0;
				Sdown[max_a_id][r] = 0;
			}
			
			aup[max_a_id] = 0;
			adown[max_a_id] = 0;
		}

		for (int max_b_id = 1; max_b_id <= maxBID; max_b_id++) 
		{
			for (int r = 1; r <= rank; r++) 
			{
				Dup[max_b_id][r] = 0;
				Ddown[max_b_id][r] = 0;
			}
			
			bup[max_b_id] = 0;
			bdown[max_b_id] = 0;
		}

		for (int max_c_id = 1; max_c_id <= maxCID; max_c_id++) 
		{
			for (int r = 1; r <= rank; r++) 
			{
				Tup[max_c_id][r] = 0;
				Tdown[max_c_id][r] = 0;
			}
			
			cup[max_c_id] = 0; 
			cdown[max_c_id] = 0;
		}

		
	}

	
	// 计算单元素的预测值
	public double getPrediction(int a_Id, int b_Id, int c_Id) {
		double p_valueHat = 0;
		for (int r = 1; r <= this.rank; r++) {	
			p_valueHat += S[a_Id][r] * D[b_Id][r] * T[c_Id][r];
		}
//		p_valueHat += a[a_Id] + b[b_Id] + c[c_Id];
		return p_valueHat; 
	}
}
