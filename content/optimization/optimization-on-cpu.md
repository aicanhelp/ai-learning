### CPU级别的优化  

 
##### 1、 循环展开  
因为处理内部针对不同的运算可能都有多个处理单元，例如，一颗CPU内部可能有4个整数加法的处理单元，优化工作就是充分利用这四个处理单元。
所谓循环展开就是，在一个循环内部把单个处理展开为相同的多个处理。例如：  

```
   int result=1;
   int len=1000;
   for (i=1;i<=len;i++){
       result=result*i;
   }
```
可以这样展开：  
```
   result1=1;
   result2=1
   int i=1;
   int len=1000;
   for (i=1;i<=len;i=i+2){
       result1=result1*i;
       result2=result2*[i+1];
   }
   
   for(;i<=len;i++){
      result1=result1*i;
   }
   
   return result1*result2
   
```

上面循环的展开的展开因子是2，通常展开因子跟CPU的操作容量C和延迟L相关，最佳展开因子等于C*L。

#### 2、消除内存引用

可以通过临时变量的方式来消除内存的引用导致的内存频繁访问。

```
   void combine1(vec_ptr v, data_t *dest){
       long i;
       long length=vec_length(v);
       data_t *data=get_vec_start(v);
       *dest=1;
       for (i=0;i<length;i++){
           *dest=*dest * i; 
       }
   }
   
   void combine2(vec_ptr v, data_t *dest){
       long i;
       long length=vec_length(v);
       data_t *data=get_vec_start(v);
       acc=1;
       for (i=0;i<length;i++){
           acc=acc * i; 
       }
       *dest=acc;
   }
```

#### 3、AVX指令的向量操作  

SIMD模型是用单条指令对整个向量进行操作，这些向量保存在特殊的向量寄存器中，向量寄存器的长度是32字节，可以存储8个32为或者4个64为数。  
所以理论上可以并行处理8组或4组数的加法或者乘法。

GCC支持向量操作。


#### 4、实现‘条件传送’

现代的CPU增加了条件传送指令，可以尝试把条件转移翻译成条件传送，这样可以避免条件预测的惩罚。因为条件传送可以被实现为普通指令的流水化处理，这样就可以条件预测的惩罚。
翻译成条件传送的基本思想是计算出条件的两个方向值，然后用条件传送选择期望的值。  
例如：  
```
   void maxmin1(long a[], long b[], long n){
       long i;
       for (i=0;i<n;i++){
           if(a[i]>b[i]){
               long t=a[i];
               a[i]=b[i];
               b[i]=t;
           }
       }
   }
   
   void maxmin2(long a[], long b[], long n){
       long i;
       for (i=0;i<n;i++){
           long min=a[i]<b[i]? a[i]:b[i];
           long max=a[i]>b[i]? b[i]:a[i];
           a[i]=min;
           b[i]=max;
       }
   }
```

对于随机数，上面的CPE（性能度量每元素周期数 Cycles Per Element）可以由13.5 降到4.0