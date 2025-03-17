import java.io.File;
import java.io.FileWriter;
import java.io.IOException;


//更新顺序U,S,T,a,b,c

 
public class SGCP extends InitTensor{

	public double sumTime = 0; //训练累计时间
	
	public int tr = 0;  //统计本轮迭代与前一轮小于误差的次数 
	
	public int threshold = 0;  //连续下降轮数小于误差范围切达到阈值终止训练
	 
	public boolean flagRMSE = true, flagMAE = true; 
	
	public String str = null;
	
	public double lambda1 = 0;  //因子矩阵正则化参数
	public double lambda2 = 0;  //线性偏差正则化参数
//	public double lambda_b = 0;  //线性偏差正则化参数
	public double tao = 0;


	
	SGCP(String trainFile, String validFile, String testFile, String separator )
	{
		super(trainFile, validFile, testFile, separator); 
	}
	
	public void train(FileWriter Parameters, int RunNo, String ProName) throws IOException
	{

		
		FileWriter  fw_RMSE = new FileWriter(new File("D:\\TensorModel\\ComparisonModel\\SGCP\\result\\"+ProName+"\\"+RunNo+"_"+"RMSE.txt"));
		FileWriter  fw_MAE = new FileWriter(new File("D:\\TensorModel\\ComparisonModel\\SGCP\\result\\"+ProName+"\\"+RunNo+"_"+"MAE.txt"));
		
		initFactorMatrix();
		initAssistMatrix();
		initSliceSet();

		System.out.println("maxAID maxBID maxCID "+maxAID+" "+maxBID+" "+maxCID);
		System.out.println("minAID minBID minCID "+minAID+" "+minBID+" "+minCID);
		System.out.println("trainCount validCount testCount "+trainCount+" "+validCount+" "+testCount);
		System.out.println("初始范围:" + initscale);
		System.out.println("lambda1  lambda2: " + lambda1 + " " + lambda2);

		long startTime = System.currentTimeMillis();   //记录开始训练时间
		int con = 1;
		for(int round = 1; round <= trainRound; round++)
		{
			initAssistMatrix();
			
			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);

				for(int r = 1; r <= rank; r++)
				{
					Sup[trainTuple.aID][r] += trainTuple.value * D[trainTuple.bID][r] * T[trainTuple.cID][r] ;
					Sdown[trainTuple.aID][r] += trainTuple.valueHat * D[trainTuple.bID][r] * T[trainTuple.cID][r] + lambda1 * trainTuple.value * (S[trainTuple.aID][r] - S[trainTuple.bID][r]) +
							lambda2 * (S[trainTuple.aID][r] + (S[trainTuple.aID][r]/Math.sqrt(Math.pow(S[trainTuple.aID][r],2) + tao)));
				}
			}
		
			for(int i = 1; i <= this.maxAID; i++)
			{
				for(int r=1; r <= rank; r++)
				{

					Sup[i][r] = (S[i][r]) * Sup[i][r];
					
//					Sdown[i][r] += lambda * aSliceSet[i] * S[i][r];
				
					
					if(Sdown[i][r] != 0)
					{
						S[i][r] = Sup[i][r] / Sdown[i][r];
		
					}
					
				}
			}
			

			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);

				for(int r = 1; r <= rank; r++)
				{
					Dup[trainTuple.bID][r] += trainTuple.value * S[trainTuple.aID][r] * T[trainTuple.cID][r];
					Ddown[trainTuple.bID][r] += trainTuple.valueHat * S[trainTuple.aID][r] * T[trainTuple.cID][r] + lambda1 * trainTuple.value * (D[trainTuple.bID][r] - D[trainTuple.aID][r]) +
							lambda2 * (D[trainTuple.bID][r] + (D[trainTuple.bID][r]/Math.sqrt(Math.pow(D[trainTuple.bID][r],2) + tao)));
				}

			}
		
			for(int j = 1; j <= this.maxBID; j++)
			{
				for(int r=1; r <= rank; r++)
				{
					Dup[j][r] = D[j][r] * Dup[j][r];
					
//					Ddown[j][r] += lambda * bSliceSet[j] * D[j][r];
			
					if(Ddown[j][r] != 0)
					{
						D[j][r] = Dup[j][r] / Ddown[j][r];
					}	
				}
			}

			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);

				for(int r = 1; r <= rank; r++)
				{
					Tup[trainTuple.cID][r] += trainTuple.value * S[trainTuple.aID][r] * D[trainTuple.bID][r];
					Tdown[trainTuple.cID][r] += trainTuple.valueHat * S[trainTuple.aID][r] * D[trainTuple.bID][r] + lambda2 * (T[trainTuple.cID][r] + (T[trainTuple.cID][r]/Math.sqrt(Math.pow(T[trainTuple.cID][r],2) + tao)));
				}
			}
		
			for(int k = 1; k <= this.maxCID; k++)
			{
				for(int r=1; r <= rank; r++)
				{
					Tup[k][r] = T[k][r] * Tup[k][r];
					
//					Tdown[k][r] += lambda * cSliceSet[k] * T[k][r];
					
					if(Tdown[k][r] != 0)
					{
						T[k][r] = Tup[k][r] / Tdown[k][r];
					}
					
				}
			}




			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);

				aup[trainTuple.aID] += trainTuple.value;
				adown[trainTuple.aID] += trainTuple.valueHat + lambda2 * a[trainTuple.aID];

			}

			for(int i = 1; i <= this.maxAID; i++)
			{
				aup[i] *= a[i];

//				adown[i] += lambda_b * aSliceSet[i] * a[i];

				if(adown[i] != 0)
				{
					a[i] = aup[i] / adown[i];
				}

			}


			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);

				bup[trainTuple.bID] += trainTuple.value;
				bdown[trainTuple.bID] += trainTuple.valueHat + lambda2 * b[trainTuple.bID];

			}

			for(int j = 1; j <= this.maxBID; j++)
			{
				bup[j] *= b[j];

//				bdown[j] += lambda_b * bSliceSet[j] * b[j];

				if(bdown[j] != 0)
				{
					b[j] = bup[j] / bdown[j];
				}

			}


			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);

				cup[trainTuple.cID] += trainTuple.value;
				cdown[trainTuple.cID] += trainTuple.valueHat + lambda2 * c[trainTuple.cID];

			}

			for(int k = 1; k <= this.maxCID; k++)
			{
				cup[k] *= c[k];

//				cdown[k] += lambda_b * cSliceSet[k] * c[k];

				if(cdown[k] != 0)
				{
					c[k] = cup[k] / cdown[k];
				}


			}

			// 每一轮参数更新后，开始对验证集测试
			double square = 0, absCount = 0;
			for (TensorTuple validTuple : validData) {
				// 获得元素的预测值
				validTuple.valueHat =this.getPrediction(validTuple.aID, validTuple.bID, validTuple.cID);
				square += Math.pow(validTuple.value - validTuple.valueHat, 2);
				absCount += Math.abs(validTuple.value - validTuple.valueHat);
				
			}
			
			everyRoundRMSE[round] = Math.sqrt(square / validCount);
			everyRoundMAE[round] = absCount / validCount; 
			
			long endRoundTime = System.currentTimeMillis();
//			sumTime += (endRoundTime-startRoundTime);
			
			// 每一轮参数更新后，记录测试集结果
			double square2 = 0, absCount2 = 0;
			for (TensorTuple testTuple : testData) {
				// 获得元素的预测值
				testTuple.valueHat = this.getPrediction(testTuple.aID, testTuple.bID, testTuple.cID);
				square2 += Math.pow(testTuple.value - testTuple.valueHat, 2);
				absCount2 += Math.abs(testTuple.value - testTuple.valueHat);
			}

			everyRoundRMSE2[round] = Math.sqrt(square2 / testCount);
			everyRoundMAE2[round] = absCount2 / testCount; 
			
//			System.out.println(round + "::" + everyRoundRMSE[round] + "::" + everyRoundMAE[round]
//					+"::"+ (endRoundTime-startRoundTime)+"::"+ everyRoundRMSE2[round] + "::" + everyRoundMAE2[round]);

			System.out.println(everyRoundRMSE2[round] + " " + everyRoundMAE2[round]);
//			fw.write(round + "::" + everyRoundRMSE[round] + "::" + everyRoundMAE[round]
//					+"::"+ (endRoundTime-startRoundTime)+"::"+ everyRoundRMSE2[round] + "::" + everyRoundMAE2[round]+"\n");
//			fw.flush();
			
			if (everyRoundRMSE[round-1] - everyRoundRMSE[round] > errorgap)
			{
				if(minRMSE > everyRoundRMSE[round])
				{
					minRMSE = everyRoundRMSE[round];
					minRMSERound = round;
				}
				
				flagRMSE = false;
				tr = 0;
			}
			
			if (everyRoundMAE[round-1] - everyRoundMAE[round] > errorgap)
			{
				if(minMAE > everyRoundMAE[round])
				{
					minMAE = everyRoundMAE[round];
					minMAERound = round;
				}
				
				flagMAE = false;
				tr = 0;
			} 
		
			if(flagRMSE && flagMAE)
			{
				tr++;
				if(tr == threshold)
				{
					con = round;
					break;
				}

			}
			
			flagRMSE = true;
			flagMAE = true;
			fw_RMSE.write(everyRoundRMSE2[round]+"\n");
			fw_MAE.write(everyRoundMAE2[round]+"\n");
			fw_RMSE.flush();
			fw_MAE.flush();
			
		}

		long endTime = System.currentTimeMillis();

//		fw.write("总训练时间："+(endTime-startTime)/1000+"s\n");
//		fw.flush();
//		fw.write("validation minRMSE:"+minRMSE+"  minRSMERound"+minRMSERound+"\n");
//		fw.write("validation minMAE:"+minMAE+"  minMAERound"+minMAERound+"\n");
//		fw.write("testing minRMSE:"+everyRoundRMSE2[minRMSERound]+"  minRSMERound"+minRMSERound+"\n");
//		fw.write("testing minMAE:"+everyRoundMAE2[minMAERound]+"  minMAERound"+minMAERound+"\n");
//		fw.flush();
//		fw.write("rank="+rank+"\n");
//		fw.flush();
//		fw.write("trainCount: "+trainCount+"validCount: "+validCount+"   testCount: "+testCount+"\n");
//		fw.flush();
		Parameters.write("testing minRMSE:" + everyRoundRMSE2[minRMSERound] + " minRSMERound::" + minRMSERound + " RMSECostTime::" + Integer.valueOf((int) ((endTime-startTime) * minRMSERound / con)) + "\n");
		Parameters.write("testing minMAE:" + everyRoundMAE2[minMAERound]+ " minMAERound::" + minMAERound + " MAECostTime::" +  Integer.valueOf((int) ((endTime-startTime) * minMAERound / con)) + "\n");
		Parameters.write( "lambda1: "+ lambda1 + " lambda2: " + lambda2  + "; 范围: " + minvalue + " to " + initscale + ";" + " \n");
		Parameters.write( "------------------------------------------------------------------------------------------------" + "\n");
		Parameters.flush();

//        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//设置日期格式
//        fw.write("训练时间："+df.format(new Date())+"\n");// new Date()为获取当前系统时间
//        fw.flush();
//		fw.write("初始范围"+initscale+"\n");
//		fw.flush();
//		fw.write("maxAID maxBID maxCID "+maxAID+" "+maxBID+" "+maxCID+"\n");
//		fw.write("minAID minBID minCID "+minAID+" "+minBID+" "+minCID+"\n");
//		fw.close();
		
		System.out.println("***********************************************");
		System.out.println("rank: "+this.rank+"\n");
//		System.out.println("validation minRMSE:"+minRMSE+"  minRSMERound"+minRMSERound);
//		System.out.println("validation minMAE:"+minMAE+"  minMAERound"+minMAERound);
//		System.out.println("总训练时间："+(endTime-startTime)/1000.00+"s\n");
		System.out.println("testing minRMSE:"+everyRoundRMSE2[minRMSERound]+"  minRSMERound"+minRMSERound);
		System.out.println("testing minMAE:"+everyRoundMAE2[minMAERound]+"  minMAERound"+minMAERound);

	}
	
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		int ProNo = 8;
		for (int number = 1; number <= ProNo; number++)
		{
			double[][] init = {{0,0}, {0.4,0.3},       {0.45,0.3},       {0.45,0.3},       {0.53,0.3},       {0.4,0.3},     {0.4,0.3},         {0.4,0.3},        {0.4,0.3}, };
			double[] lambda1 = {0, Math.pow(2, -7), Math.pow(2, -7), Math.pow(2, -6), Math.pow(2, -4), Math.pow(2, -8), Math.pow(2, -7), Math.pow(2, -6), Math.pow(2, -7)};
			double[] lambda2 = {0, Math.pow(2, -10), Math.pow(2, -11), Math.pow(2, -11), Math.pow(2, -8), Math.pow(2, -11), Math.pow(2, -11), Math.pow(2, -11), Math.pow(2, -11)};
			int RunNo = 3;
			String ProName = "BTC-"+ number;
			for (int i = 1; i <= RunNo; i++) {
				FileWriter Parameters = new FileWriter(new File("D:\\TensorModel\\ComparisonModel\\SGCP\\result\\" + ProName + "\\" + i + ".txt"));
				SGCP bnlft = new SGCP(
						"C:\\Users\\Administrator\\Desktop\\Data\\BTC2(sigmoid)1_2_7\\" + number + ".1\\train_0.1.txt",
						"C:\\Users\\Administrator\\Desktop\\Data\\BTC2(sigmoid)1_2_7\\" + number + ".1\\val_0.2.txt",
						"C:\\Users\\Administrator\\Desktop\\Data\\BTC2(sigmoid)1_2_7\\" + number + ".1\\test_0.7.txt", "::");
				bnlft.initscale = init[number][0];
				bnlft.minvalue = init[number][1];
				bnlft.rank = 20;
				bnlft.trainRound = 1000;
				bnlft.errorgap = 1E-5;
				bnlft.lambda1 = lambda1[number];
				bnlft.lambda2 = lambda2[number];
				bnlft.tao = Math.pow(10, (-8));

				try {
					bnlft.initData(bnlft.trainFile, bnlft.trainData, 1);
					bnlft.initData(bnlft.validFile, bnlft.validData, 2);
					bnlft.initData(bnlft.testFile, bnlft.testData, 3);
					bnlft.train(Parameters, i, ProName);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}
}

