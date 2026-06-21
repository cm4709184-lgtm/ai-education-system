const int sampleWindow = 50; // 以mS为单位的采样窗口宽度（50 mS = 20Hz）  

unsigned int sample;

void setup() 

{

  Serial.begin(9600);

  pinMode(A0,INPUT); 

}

void loop() 

{

  unsigned long startMillis= millis(); // 样本窗口的开始 

  unsigned int peakToPeak = 0;  // 峰峰值

  unsigned int signalMax = 0;

  unsigned int signalMin = 1024;

  // collect data for 50 mS

  while (millis() - startMillis < sampleWindow)

  {

   sample = analogRead(A0);

   if (sample < 1024) // 抛出错误的读数

   {

     if (sample > signalMax)

     {

      signalMax = sample; // 只保存最大级别

     }

     else if (sample < signalMin)

     {

      signalMin = sample; // 仅保存最低级别

     }

   }

  }

  peakToPeak = signalMax - signalMin; // max-min =峰峰值幅度

  double volts = (peakToPeak * 5.0) / 1024; // 转换为伏特

  Serial.println(volts);

}