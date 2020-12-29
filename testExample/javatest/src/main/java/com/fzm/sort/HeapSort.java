package com.fzm.sort;

/**
 * @ProjectName: testExample
 * @Package: com.fzm.sort
 * @ClassName: HeapSort
 * @description: 堆排序
 * @Author: fangzhimeng
 * @CreateDate: 2020/12/1 15:08
 * @UpdateUser: 更新者
 * @UpdateDate: 2020/12/1 15:08
 * @UpdateRemark: 更新说明
 * @Version: 1.0
 * @create: 2020-12-01 15:08
 */
public class HeapSort {
    public static void heapSort(int[] array){
        buildHeap(array);
        int n = array.length;
        int i=0;
        //取出该最大堆的根节点，同时，取最末尾的叶子节点来作为根节点，从此根节点开始调整堆，使其满足最大堆的特性
        //直到堆的大小由n个元素降到2个
        for(i=n-1;i>=1;i--){
            swap(array,0,i);
            heapify(array,0,i);
            for (int j = 0; j < array.length; j++) {
                System.out.print(array[j]);
                System.out.print(",");
            }
            System.out.println();
        }
    }

    //构建堆
    public static void buildHeap(int[] array){
        for(int i=array.length/2-1;i>=0;i--){
            heapify(array,i,array.length);
        }
    }

    //调整堆
    public static void heapify(int[] data,int parentNode,int heapSize){
        int leftChild = 2*parentNode+1;// 左子树的下标
        int rightChild =2*parentNode+2;// 右子树的下标（如果存在的话）
        int largest = 0;
        //寻找3个节点中最大值节点的下标
        if(leftChild<heapSize && data[leftChild]>data[parentNode]){
            largest = leftChild;
        }else if(rightChild<heapSize && data[rightChild]>data[largest]){
            largest = rightChild;
        }else{
            largest = parentNode;
        }
        //如果最大值不是父节点，那么交换，并继续调整堆
        if(largest!=parentNode){
            swap(data,largest,parentNode);
            heapify(data,largest,heapSize);
        }
    }
    //交换函数
    public static void swap(int[] array,int i,int j){
        int temp =0;
        temp=array[i];
        array[i]=array[j];
        array[j]=temp;
    }
    public static void main(String[] args) {
        int[] arr = { 55, 56, 23, 90, 47, 9, 40, 82, 76, 33 };
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i]);
            System.out.print(",");
        }
        System.out.println();
        heapSort(arr);
    }
}

