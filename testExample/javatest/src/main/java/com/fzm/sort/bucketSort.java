package com.fzm.sort;

/**
 * @ProjectName: testExample
 * @Package: com.fzm.sort
 * @ClassName: bucketSort
 * @description: 桶排序
 * @Author: fangzhimeng
 * @CreateDate: 2020/12/1 14:40
 * @UpdateUser: 更新者
 * @UpdateDate: 2020/12/1 14:40
 * @UpdateRemark: 更新说明
 * @Version: 1.0  https://cloud.tencent.com/developer/article/1423184
 * @create: 2020-12-01 14:40
 */
public class bucketSort {
    public static void bucketSort(int[] arr) {

        if (arr == null || arr.length < 2) {
            return;
        }

        int max = Integer.MIN_VALUE;

        for (int i = 0; i < arr.length; i++) {
            max = Math.max(max, arr[i]);
        }

        int[] bucket = new int[max + 1];

        for (int i = 0; i < arr.length; i++) {
            bucket[arr[i]]++;
        }

        int i = 0;

        for (int j = 0; j < bucket.length; j++) {
            while (bucket[j]-- > 0) {
                arr[i++] = j;
            }
        }
    }
}
