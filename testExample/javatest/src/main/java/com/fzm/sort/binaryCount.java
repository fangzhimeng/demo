package com.fzm.sort;

/**
 * @ProjectName: testExample
 * @Package: com.fzm.sort
 * @ClassName: binaryCount
 * @description: 二进制1的个数
 * @Author: fangzhimeng
 * @CreateDate: 2020/12/2 10:33
 * @UpdateUser: 更新者
 * @UpdateDate: 2020/12/2 10:33
 * @UpdateRemark: 更新说明
 * @Version: 1.0
 * @create: 2020-12-02 10:33
 */
public class binaryCount {
    public static void main(String[] args) {
        NumberOf1(10);
    }

    public static int NumberOf1(int n) {
        int cnt = 0;
        while (n != 0) {
            cnt++;
            n &= (n - 1);
        }
        return cnt;
    }
}
