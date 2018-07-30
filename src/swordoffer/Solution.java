package swordoffer;

import java.lang.reflect.Array;
import java.util.*;

public class Solution {

    // 树的根节点
    private class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    // 计算二叉树的最浅深度
    public int findMinDepthForBinaryTree(TreeNode root) {
        if (root == null)
            return 0;
        int leftMinDepth = findMinDepthForBinaryTree(root.left);
        int rightMinDepth = findMinDepthForBinaryTree(root.right);
        // 深度为0表示没有子节点
        if (leftMinDepth == 0 || rightMinDepth == 0)
            return 1 + leftMinDepth + rightMinDepth;
        return 1 + Math.min(leftMinDepth, rightMinDepth);
    }


    // 利用栈来实现逆波兰解析
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<Integer>();
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i].equals("+") || tokens[i].equals("-")
                    || tokens[i].equals("*") || tokens[i].equals("/")) {
                int b = stack.pop();
                int a = stack.pop();
                stack.push(cal(tokens[i], a, b));
            } else {
                stack.push(Integer.parseInt(tokens[i]));
            }
        }
        return stack.pop();
    }

    private int cal(String operator, int a, int b) {
        if (operator.equals("+"))
            return a + b;
        if (operator.equals("-"))
            return a - b;
        if (operator.equals("*"))
            return a * b;
        return a / b;
    }

    // 1. 剑指offer：二位数组中的查找
    public boolean Find(int target, int[][] array) {
        int i = 0;
        int j = array.length - 1;
        while (i < array[0].length && j >= 0) {
            if (array[i][j] < target) {
                i++;
            } else if (array[i][j] > target) {
                j--;
            } else {
                return true;
            }
        }
        return false;
    }

    // 2. 剑指offer：替换空格
    public String replaceSpace(StringBuffer str) {
        StringBuffer replaceStr = new StringBuffer();
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == ' ')
                replaceStr.append("%20");
            else
                replaceStr.append(str.charAt(i));
        }
        return String.valueOf(replaceStr);
    }

    // 3. 剑指offer：从尾到头打印单链表元素
    public class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> arrayList = new ArrayList<Integer>();
        Stack<Integer> stack = new Stack<>();
        ListNode node = listNode;
        while (node != null) {
            stack.push(node.val);
            node = node.next;
        }
        while (!stack.isEmpty()) {
            arrayList.add(stack.pop());
        }
        return arrayList;
    }

    // 4. 剑指offer：重建二叉树
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {

        if (pre.length == 0 || in.length == 0)
            return null;
        TreeNode node = new TreeNode(pre[0]);
        for (int i = 0; i < in.length; i++) {
            if (in[i] == pre[0]) {
                node.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, i + 1),
                        Arrays.copyOfRange(in, 0, i));
                node.right = reConstructBinaryTree(Arrays.copyOfRange(pre, i + 1, pre.length),
                        Arrays.copyOfRange(in, i + 1, in.length));
            }
        }
        return node;
    }

    // 5. 剑指offer：用两个栈实现队列
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

    public void push1(int node) {
        stack1.push(node);
    }

    public int pop1() {
        if (stack2.empty()){
            while (!stack1.empty()){
                stack2.push(stack1.pop());
            }
        }
        return stack2.pop();
    }

    // 6. 剑指offer：旋转数组的最小数字
    public int minNumberInRotateArray(int [] array) {
        if (array.length <= 0)
            return 0;
        int left = 0;
        int right = array.length -1 ;
        int middle;

        while (true){
            if (right - left == 1)
                return array[right];
            middle = (left + right) / 2;
            if (array[middle] >= array[right])
                left = middle;
            else
                right = middle;
        }
    }

    // 7. 剑指offer：矩阵覆盖
    public int RectCover(int target) {
        if (target == 0)
            return 0;
        if (target == 1)
            return 1;
        if (target == 2)
            return 2;
        int result = RectCover(target - 1) + RectCover(target - 2);
        return result;
    }

    // 8. 剑指offer：二进制中1的个数（位运算）
    public int NumberOf1(int n) {
        int count = 0;
        while (n != 0){
            count++;
            n = n & (n - 1);
        }
        return count;
    }

    // 9. 剑指offer：数值的整数次方
    public double Power(double base, int exponent) {
        return Math.pow(base, exponent);
    }

    // 10. 剑指offer：调整数组顺序，使奇数位于偶数前边
    public void reOrderArray(int [] array) {
        ArrayList<Integer> arrayList1 = new ArrayList<>();
        ArrayList<Integer> arrayList2 = new ArrayList<>();

        for (int i = 0; i < array.length; i++){
            if ((array[i] & 1) == 1)
                arrayList1.add(array[i]);
            else
                arrayList2.add(array[i]);
        }

        int index = 0;
        for (Integer value : arrayList1){
            array[index++] = value;
        }
        for (Integer value : arrayList2){
            array[index++] = value;
        }
    }

    // 11. 剑指offer：链表中倒数第k个节点
    public ListNode FindKthToTail(ListNode head,int k) {
        if (head == null)
            return null;
        ListNode node = head;
        int length = 0;
        while (node != null){
            length++;
            node = node.next;
        }

        if(k > length)
            return null;
        node = head;
        for (int i = 1; i <= length - k; i++) {
            node = node.next;
        }
        return node;
    }

    // 12. 剑指offer：反转链表
    public ListNode ReverseList(ListNode head) {
        if (head == null) return head;
        ListNode node = head;
        ListNode pre = null;
        ListNode next;
        while (node != null){
            next = node.next;
            node.next = pre;
            pre = node;
            node = next;
        }
        head = pre;
        return head;
    }

    // 13. 剑指offer：合并两个排序的链表
    public ListNode Merge(ListNode list1,ListNode list2) {

        if (list1 == null) return list2;
        else if (list2 == null) return list1;

        ListNode head;
        if (list1.val < list2.val){
            head = list1;
            list1 = list1.next;
        }
        else {
            head = list2;
            list2 = list2.next;
        }

        ListNode node = head;
        while (list1 != null && list2 != null){
            if (list1.val < list2.val){
                node.next = list1;
                list1 = list1.next;
                node = node.next;
            }
            else {
                node.next = list2;
                list2 = list2.next;
                node = node.next;
            }
        }

        while (list1 != null){
            node.next = list1;
            list1 = list1.next;
            node = node.next;
        }

        while (list2 != null){
            node.next = list2;
            list2 = list2.next;
            node = node.next;
        }

        return head;
    }

    // 14. 剑指offer：树的子结构
    public boolean HasSubtree(TreeNode root1,TreeNode root2) {
        boolean result = false;
        if (root1 != null && root2 != null) {
            if (root1.val == root2.val)
                result = judgeIfSubTree(root1, root2);
            if (!result) result = HasSubtree(root1.left, root2);
            if (!result) result = HasSubtree(root1.right, root2);
        }
        return result;
    }

    public boolean judgeIfSubTree(TreeNode root1, TreeNode root2){
        if (root2 == null) return true;
        if (root1 == null) return false;
        if (root1.val != root2.val) return false;
        return judgeIfSubTree(root1.left, root2.left) && judgeIfSubTree(root1.right, root2.right);
    }

    // 15. 剑指offer：二叉树的镜像
    public void Mirror(TreeNode root) {
        Reverse(root);
    }

    public TreeNode Reverse(TreeNode root){
        if (root == null)
            return null;
        TreeNode tmpNode = root.left;
        root.left = Reverse(root.right);
        root.right = Reverse(tmpNode);
        return root;
    }

    // 16. 剑指offer：顺时针打印矩阵
    public ArrayList<Integer> printMatrix(int [][] array) {
        ArrayList<Integer> result = new ArrayList<> ();
        if(array.length == 0)
            return result;
        int n = array.length, m = array[0].length;
        if(m == 0) return result;
        int layers = (Math.min(n,m) - 1) / 2 + 1;//这个是层数
        for(int i = 0; i < layers; i++){
            for(int k = i; k < m - i; k++)
                result.add(array[i][k]);//左至右
            for(int j = i + 1; j < n - i; j++)
                result.add(array[j][m - i - 1]);//右上至右下
            for(int k = m - i - 2; k >= i && n - i - 1 != i; k--)
                result.add(array[n - i - 1][k]);//右至左
            for(int j= n - i - 2; j > i && m - i - 1 != i ;j--)
                result.add(array[j][i]);//左下至左上
        }
        return result;
    }

    // 17. 剑指offer：包含min函数的栈
    Stack<Integer> dataStack = new Stack<>();
    Stack<Integer> minStack = new Stack<>();
    int min = Integer.MAX_VALUE;

    public void push(int node) {
        dataStack.push(node);
        if (node < min){
            minStack.push(node);
            min = node;
        }
        else
            minStack.push(min);
    }

    public void pop() {
        dataStack.pop();
        minStack.pop();
        min = minStack.peek();
    }

    public int top() {
        return dataStack.peek();
    }

    public int min() {
        return min;
    }

    // 18. 剑指offer：栈的压入、弹出序列
    public boolean IsPopOrder(int [] pushA,int [] popA) {
        Stack<Integer> tmpStack = new Stack<>();
        int countPop = 0;
        for (int i = 0; i < pushA.length; i++){
            tmpStack.push(pushA[i]);
            for (int j = countPop; j < popA.length; j++) {
                if (tmpStack.peek() == popA[j]){
                    tmpStack.pop();
                    countPop++;
                }
                else
                    break;
            }
        }
        return tmpStack.empty();
    }

    // 19. 剑指offer：从上往下打印二叉树
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> arrayList = new ArrayList<>();
        if (root == null) return arrayList;
        ArrayList<TreeNode> queue = new ArrayList<>();
        TreeNode node = root;
        queue.add(node);
        while (!queue.isEmpty()){
            node = queue.remove(0);
            arrayList.add(node.val);
            if (node.left != null) queue.add(node.left);
            if (node.right != null) queue.add(node.right);
        }
        return arrayList;
    }

    // 20. 剑指offer：二叉搜索树的后序遍历序列
    public boolean VerifySquenceOfBST(int [] sequence) {
        if (sequence.length == 0)
            return false;
        return VerifySquenceIsBST(sequence);
    }

    private boolean VerifySquenceIsBST(int [] sequence) {
        if (sequence.length == 0)
            return true;
        int pivot = sequence[sequence.length - 1];
        int boundary = 0;
        while (sequence[boundary] < pivot){
            boundary++;
        }
        for (int i = boundary; i < sequence.length - 1; i++) {
            if (sequence[i] < pivot)
                return false;
        }
        return VerifySquenceIsBST(Arrays.copyOfRange(sequence, 0, boundary)) &
                VerifySquenceIsBST(Arrays.copyOfRange(sequence, boundary, sequence.length - 1));
    }

    // 21. 剑指offer：二叉树中和为某一值的路径
    private ArrayList<ArrayList<Integer>> allPath = new ArrayList<>();
    private ArrayList<Integer> path = new ArrayList<>();
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
        if (root == null) return allPath;
        path.add(root.val);
        target -= root.val;
        if (target == 0 && root.left == null && root.right == null)
            allPath.add(new ArrayList<>(path));

        FindPath(root.left, target);
        FindPath(root.right, target);

        path.remove(path.size() - 1);
        return allPath;
    }

    // 22. 剑指offer：复杂链表的复制
    public class RandomListNode {
        int label;
        RandomListNode next = null;
        RandomListNode random = null;

        RandomListNode(int label) {
            this.label = label;
        }
    }

    public RandomListNode Clone(RandomListNode pHead)
    {
        if (pHead == null) return pHead;
        RandomListNode node = pHead;
        while (node != null){
            RandomListNode newNode = new RandomListNode(node.label);
            newNode.next = node.next;
            node.next = newNode;
            node = newNode.next;
        }
        node = pHead;
        while (node != null){
            if (node.random != null)
                node.next.random = node.random.next;
            node = node.next.next;
        }

        node = pHead;
        RandomListNode newNode = pHead.next;
        RandomListNode newHead = newNode;

        while (node != null){
            node.next = node.next.next;
            if(node.next != null){
                newNode.next = newNode.next.next;
                newNode = newNode.next;
            }
            node = node.next;
        }

        return newHead;
    }

    // 23. 剑指offer：二叉搜索树与双向链表
    private TreeNode leftNode = null, rightNode = null;
    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null) return null;
        Convert(pRootOfTree.left);
        if (rightNode == null)
            rightNode = leftNode = pRootOfTree;
        else {
            rightNode.right = pRootOfTree;
            pRootOfTree.left = rightNode;
            rightNode = pRootOfTree;
        }
        Convert(pRootOfTree.right);
        return leftNode;
    }

    // 24. 剑指offer：字符串的排列
    public ArrayList<String> Permutation(String str) {
        ArrayList<String> res = new ArrayList<>();
        if (str != null && str.length() > 0) {
            PermutationHelper(str.toCharArray(), 0, res);
            Collections.sort(res);
        }
        return res;
    }

    private void PermutationHelper(char[] cs, int i, ArrayList<String> list) {
        if (i == cs.length - 1) {
            String val = String.valueOf(cs);
            if (!list.contains(val))
                list.add(val);
        } else {
            for (int j = i; j < cs.length; j++) {
                swap(cs, i, j);
                PermutationHelper(cs, i + 1, list);
                swap(cs, i, j);
            }
        }
    }

    private void swap(char[] cs, int i, int j) {
        char temp = cs[i];
        cs[i] = cs[j];
        cs[j] = temp;
    }

    // 25. 剑指offer：数组中出现次数超过一半的数字
    public int MoreThanHalfNum_Solution(int [] array) {
        int result = 0;
        if (array.length == 0) return result;
        int count = 0;
        for (int anArray : array) {
            if (count == 0) {
                result = anArray;
                count++;
            } else {
                if (anArray == result) count++;
                else count--;
            }
        }

        count = 0;
        for (int anArray : array) {
            if (anArray == result) count++;
        }

        if (count * 2 > array.length) return result;
        else return 0;
    }

    // 26. 剑指offer：最小的k个数
    public static ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
        ArrayList<Integer> result = new ArrayList<>();
        if (input.length == 0 || input.length < k) return result;
        if (k == input.length){
            for (int val : input) result.add(val);
            return result;
        }
        else {
            quickSelect(input, 0, input.length - 1, k);
            for (int i = 0; i < k; i++) result.add(input[i]);
            return result;
        }
    }

    private static void swap(int[] array, int i, int j) {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    private static int median3(int[] array, int left, int right){
        int center = (left + right) / 2;
        if (array[left] > array[center]) swap(array, left, center);
        if (array[left] > array[right]) swap(array, left, right);
        if (array[center] > array[right]) swap(array, center, right);
        swap(array, center, right - 1);
        return array[right - 1];
    }

    private static void quickSelect(int[] array, int left, int right, int k){
        int pivot = median3(array, left, right);
        int i = left;
        int j = right;
        while (true){
            while (array[i] < pivot) i++;
            while (array[j] >= pivot) j--;
            if (i < j) swap(array, i, j);
            else break;
        }
        swap(array, i, right - 1);
        if (i < k - 1) quickSelect(array, i + 1, right, k);
        else if (i > k - 1 && i != 1) quickSelect(array, left, i - 1, k);
    }

    // 27. 剑指offer：连续子数组的最大和
    public int FindGreatestSumOfSubArray(int[] array) {
        if (array.length == 1) return array[0];
        int maxVal = Integer.MIN_VALUE;
        int count;
        for (count = 0; count < array.length; count++){
            if (array[count] > 0)
                break;
            else if (array[count] > maxVal)
                maxVal = array[count];
        }
        if (count == array.length)
            return maxVal;
        else {
            int sum = 0;
            for (int anArray : array) {
                sum += anArray;
                if (sum <= 0)
                    sum = 0;
                if (sum > maxVal)
                    maxVal = sum;
            }
            return maxVal;
        }
    }

    // 28. 剑指offer：整数中1出现的次数（1~n中1出现的次数）
    public int NumberOf1Between1AndN_Solution(int n) {
        int result = 0;
        for (int i = 1; i <= n; i++) {
            String str = String.valueOf(i);
            char[] cs = str.toCharArray();
            for (char c : cs) {
                if (c == '1')
                    result++;
            }
        }
        return result;
    }

    // 29. 剑指offer：把数组排成最小的数
    public String PrintMinNumber(int [] numbers) {
        if (numbers.length == 0) return "";
        ArrayList<Integer> arrayList = new ArrayList<>();
        for (int val : numbers){
            arrayList.add(val);
        }

        arrayList.sort((o1, o2) -> {
            String str1 = o1 + "" + o2;
            String str2 = o2 + "" + o1;
            return str1.compareTo(str2);
        });

        StringBuilder result = new StringBuilder();
        for (Integer val : arrayList){
            result.append(val);
        }
        return String.valueOf(result);
    }

    // 30. 剑指offer：丑数
    public int GetUglyNumber_Solution(int index) {
        if (index <= 0) return 0;
        int[] result = new int[index];
        int count = 0;
        int i2 = 0;
        int i3 = 0;
        int i5 = 0;

        result[0] = 1;

        int tmp = 0;
        while (count < index - 1){
            tmp = Math.min(result[i2] * 2, Math.min(result[i3] * 3, result[i5] * 5));
            if (tmp == result[i2] * 2) i2++;
            if (tmp == result[i3] * 3) i3++;
            if (tmp == result[i5] * 5) i5++;
            result[++count] = tmp;
        }

        return result[index - 1];
    }

    // 31. 剑指offer：第一个只出现一次的字符
    public int FirstNotRepeatingChar(String str) {
        if (str == null) return -1;
        HashMap<Character, Integer> map = new HashMap<>(str.length());
        for (int i = 0; i < str.length(); i++) {
            char key = str.charAt(i);
            if (!map.containsKey(key))
                map.put(key, 1);
            else {
                int oldVal = map.get(key);
                oldVal++;
                map.put(key, oldVal);
            }
        }
        for (int i = 0; i < str.length(); i++) {
            if (map.get(str.charAt(i)) == 1)
                return i;
        }
        return -1;
    }

    // 32. 剑指offer：数组中的逆序对
    public int InversePairs(int [] array) {
        if (array.length == 0) return 0;
        int[] copy = new int[array.length];
        return InversePairsCore(array, copy, 0, array.length - 1);
    }

    private int InversePairsCore(int[] array, int[] copy, int left, int right){
        if (left == right) return 0;
        int middle = (left + right) / 2;
        int leftCount = InversePairsCore(array, copy, left, middle) % 1000000007;
        int rightCount = InversePairsCore(array, copy, middle + 1, right) % 1000000007;
        int count = 0;
        int i = middle;
        int j = right;
        int tmpCopy = right;
        while (i >= left && j > middle){
            if (array[i] > array[j]){
                copy[tmpCopy--] = array[i--];
                count += j - middle;
                if (count > 1000000007) count %= 1000000007;
            }
            else {
                copy[tmpCopy--] = array[j--];
            }
        }
        while (i >= left){
            copy[tmpCopy--] = array[i--];
        }
        while (j > middle){
            copy[tmpCopy--] = array[j--];
        }
        for (int k = left; k <= right; k++){
            array[k] = copy[k];
        }
        return (leftCount + rightCount + count) % 1000000007;
    }

    // 33. 剑指offer：两个链表的第一个公共节点
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if (pHead1 == null || pHead2 == null) return null;
        ListNode node1 = pHead1;
        ListNode node2 = pHead2;
        Stack<ListNode> stack1 = new Stack<>();
        Stack<ListNode> stack2 = new Stack<>();
        while (node1 != null){
            stack1.push(node1);
            node1 = node1.next;
        }
        while (node2 != null){
            stack2.push(node2);
            node2 = node2.next;
        }
        ListNode commonNode = null;
        while (!stack1.isEmpty() && !stack2.isEmpty()){
            if (stack1.peek() == stack2.peek()){
                commonNode = stack1.pop();
                stack2.pop();
            }
            else {
                return commonNode;
            }
        }
        return commonNode;
    }

    // 34. 剑指offer：数字在排序数组中出现的次数
    public static int GetNumberOfK(int [] array , int k) {
        if (array == null || array.length == 0) return -1;
        int pos = getPosition(array, k, 0, array.length - 1);
        int result = 0;
        int star = pos, end = pos + 1;
        while (star >= 0 && array[star] == k){
            result++;
            star--;
        }
        while (end < array.length && array[end] == k){
            result++;
            end++;
        }
        return result;
    }

    private static int getPosition(int[] array, int k, int left, int right){
        if (left > right) return -1;
        int middle = (left + right) >> 1;
        if (array[middle] < k)
            return getPosition(array, k, middle + 1, left);
        if (array[middle] > k)
            return getPosition(array, k, left, middle - 1);
        else
            return middle;
    }

    // 35. 剑指offer：二叉树的深度
    public int TreeDepth(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.max(TreeDepth(root.left), TreeDepth(root.right));
    }

    // 36. 剑指offer：平衡二叉树
    public boolean IsBalanced_Solution(TreeNode root) {
        return heightAVL(root) != -1;
    }
    private int heightAVL(TreeNode root){
        if (root == null) return 0;
        int left = heightAVL(root.left);
        if (left == -1) return -1;
        int right = heightAVL(root.right);
        if (right == -1) return -1;
        if (Math.abs(left - right) <= 1)
            return 1 + Math.max(left, right);
        else
            return -1;
    }

    // 37. 剑指offer：数组中只出现一次的数字
    public void FindNumsAppearOnce(int [] array,int num1[] , int num2[]) {
        int x = array[0];
        for(int i = 1; i < array.length; i++)
            x ^= array[i];
        int flag = 1;
        while ((x & flag) == 0)
            flag = flag << 1;
        for (int val : array){
            if ((val & flag) == flag)
                num1[0] ^= val;
            else
                num2[0] ^= val;
        }
    }

    // 38. 和为S的连续正数序列
    public static ArrayList<ArrayList<Integer> > FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> allList = new ArrayList<>();
        ArrayList<Integer> sequence;
        if (sum <= 1) return allList;
        for (int n = (int)Math.round((double)sum / 2); n > 1; n--) {
            double compare = (double) (sum - (n * (n - 1)) / 2) / n;
            int x = (sum - (n * (n - 1)) / 2) / n;
            if (x == compare && x > 0){
                sequence = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    sequence.add(x++);
                }
                allList.add(sequence);
            }
        }
        return allList;
    }

    // 39. 剑指offer：和为S的两个数字
    public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {
        ArrayList<Integer> result = new ArrayList<>();
        if (array.length == 0 || array.length == 1) return result;
        int i = 0, j = array.length - 1;
        while (i < j){
            if (array[i] + array[j] == sum){
                result.add(array[i]);
                result.add(array[j]);
                return result;
            }
            else if (array[i] + array[j] > sum) j--;
            else i++;
        }
        return result;
    }

    // 40. 剑指offer：左旋转字符串
    public String LeftRotateString(String str,int n) {
//        if (str.length() == 0) return null;
//        int i = 0, j = n - 1;
//        char[] cs = str.toCharArray();
//        while (i < j){
//            char tmp = cs[i];
//            cs[i] = cs[j];
//            cs[j] = tmp;
//        }
//        i = n;
//        j = str.length() - 1;
//        while (i < j){
//            char tmp = cs[i];
//            cs[i] = cs[j];
//            cs[j] = tmp;
//        }
//        i = 0;
//        j = str.length() - 1;
//        while (i < j){
//            char tmp = cs[i];
//            cs[i] = cs[j];
//            cs[j] = tmp;
//        }
//        return String.valueOf(cs);
        if (str.length() == 0) return "";
        StringBuilder stringB = new StringBuilder(str);
        String tmp = String.valueOf(stringB.substring(0, n));
        stringB.delete(0, n);
        stringB.append(tmp);
        return String.valueOf(stringB);
    }

    // 41. 剑指offer：翻转单词顺序列
    public String ReverseSentence(String str) {
        if (str.length() == 0) return "";
        if (str.trim().equals("")) return str;
        String[] strArray = str.split(" ");
        int i = 0, j = strArray.length - 1;
        while (i < j){
            String tmp = strArray[i];
            strArray[i] = strArray[j];
            strArray[j] = tmp;
            i++;
            j--;
        }
        StringBuilder result = new StringBuilder();
        for (String val : strArray){
            result.append(val);
            result.append(" ");
        }
        return String.valueOf(result).trim();
    }

    // 42. 剑指offer：扑克牌顺子
    public boolean isContinuous(int [] numbers) {
        if(numbers.length != 5) return false;
        ArrayList<Integer> judeRe = new ArrayList<>();
        int max = -1;
        int min = 14;
        for (int val : numbers) {
            if (val < 0 || val > 13) return false;
            if (val == 0) continue;
            if (judeRe.contains(val)) return false;
            judeRe.add(val);
            if (val < min) min = val;
            else if (val > max) max = val;
            if ((max - min) > 4) return false;
        }
        return true;
    }

    // 43. 剑指offer：孩子们的游戏（圆圈中最后剩下的数）
    public int LastRemaining_Solution(int n, int m) {
        if (n <= 0) return -1;
        LinkedList<Integer> linkedList = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            linkedList.add(i);
        }
        int begin = 0;
        while (linkedList.size() > 1){
            begin = (begin + m - 1) % linkedList.size();
            linkedList.remove(begin);
        }
        return linkedList.get(0);
    }

    // 44. 剑指offer：求1+2+3+...+n不能用乘除法和循环和if
    public int Sum_Solution(int n) {
        int sum = n;
        boolean ans = (n != 0) && (sum += Sum_Solution(n - 1)) > 0;
        return sum;
    }

    // 45. 剑指offer：不用加减乘除做加法
    public int Add(int num1,int num2) {
        while(num2 != 0){
            int sum = num1 ^ num2;
            num2 = (num1 & num2) << 1;
            num1 = sum;
        }
        return num1;
    }

    // 46. 剑指offer：把字符串转换成整数
    public int StrToInt(String str) {
        if (str.equals("") || str.length() == 0) return 0;
        char[] sArray = str.toCharArray();
        int symbol = 1, i = 0;
        if (sArray[0] == '-'){
            symbol = 0;
            i++;
        }
        if (sArray[0] == '+'){
            i++;
        }
        int sum = 0;
        for (; i < sArray.length; i++) {
            if (sArray[i] < 48 || sArray[i] > 57) return 0;
            sum = sum * 10 + sArray[i] - 48;
        }
        if (symbol == 1) return sum;
        else return sum * -1;
    }

    // 46. 剑指offer：数组中重复的数字
    // Parameters:
    //    numbers:     an array of integers
    //    length:      the length of array numbers
    //    duplication: (Output) the duplicated number in the array number,length of duplication array is 1,so using duplication[0] = ? in implementation;
    //                  Here duplication like pointor in C/C++, duplication[0] equal *duplication in C/C++
    //    这里要特别注意~返回任意重复的一个，赋值duplication[0]
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
    public boolean duplicate(int numbers[],int length,int [] duplication) {
        if (length <= 1) return false;
        ArrayList<Integer> arrayList = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            if (arrayList.contains(numbers[i])){
                duplication[0] = numbers[i];
                return true;
            }
            else {
                arrayList.add(numbers[i]);
            }
        }
        return false;
    }

    // 47. 剑指offer：构建乘积数组
    public int[] multiply(int[] A) {
        int[] B = new int[A.length];
        if (A.length <= 1) return B;
        B[0] = 1;
        for (int i = 1; i < A.length; i++){
            B[i] = B[i - 1] * A[i - 1];
        }
        int tmp = 1;
        for (int i = A.length -1; i >= 0; i--){
            B[i] = B[i] * tmp;
            tmp = tmp * A[i];
        }
        return B;
    }

    // 48. 剑指offer：正则表达式匹配
    public boolean match(char[] str, char[] pattern)
    {
        if (str == null || pattern == null) return false;
        return matchCore(str, 0, pattern, 0);
    }

    private boolean matchCore(char[] str, int strIndex,
                              char[] pattern, int patternIndex){
        if (strIndex == str.length &&
                patternIndex == pattern.length){
            return true;
        }

        if (strIndex != str.length &&
                patternIndex == pattern.length)
            return false;

        if (patternIndex + 1 < pattern.length &&
                pattern[patternIndex + 1] == '*'){
            if ((strIndex != str.length && pattern[patternIndex] == str[strIndex]) ||
                    (pattern[patternIndex] == '.' && strIndex != str.length)){
                return matchCore(str, strIndex, pattern, patternIndex + 2)
                        || matchCore(str, strIndex + 1, pattern, patternIndex + 2)
                        || matchCore(str, strIndex + 1, pattern, patternIndex);
            }
            else {
                return matchCore(str, strIndex, pattern, patternIndex + 2);
            }
        }

        if ((strIndex != str.length) && pattern[patternIndex] == str[strIndex] ||
                (pattern[patternIndex] == '.' && strIndex != str.length)){
            return matchCore(str, strIndex + 1, pattern, patternIndex + 1);
        }

        return false;
    }

    // 49. 剑指offer：显示数值的字符串
    public boolean isNumeric(char[] str) {
        String string = String.valueOf(str);
        return string.matches("[+-]?[0-9]*(\\.[0-9]*)?([eE][+-]?[0-9]+)?");
    }

    // 50. 剑指offer：字符流中第一个不重复的字符
    int[] hashTable = new int[256];
    StringBuilder stringBuilder = new StringBuilder();
    //Insert one char from stringstream
    public void Insert(char ch)
    {
        stringBuilder.append(ch);
        hashTable[ch] += 1;
    }
    //return the first appearence once char in current stringstream
    public char FirstAppearingOnce()
    {
        char[] cs = String.valueOf(stringBuilder).toCharArray();
        for (char val : cs){
            if (hashTable[val] == 1)
                return val;
        }
        return '#';
    }

    // 51. 剑指offer：按之字形顺序打印二叉树
    public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
        int layer = 1;
        Stack<TreeNode> s1 = new Stack<>();
        s1.push(pRoot);
        Stack<TreeNode> s2 = new Stack<>();

        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        while (!s1.empty() || !s2.empty()){
            if ((layer & 1) == 1){
                ArrayList<Integer> tmp = new ArrayList<>();
                while (!s1.empty()){
                    TreeNode node = s1.pop();
                    if (node != null){
                        tmp.add(node.val);
                        s2.push(node.left);
                        s2.push(node.right);
                    }
                }
                if (!tmp.isEmpty()){
                    result.add(tmp);
                    layer++;
                }
            }
            else {
                ArrayList<Integer> tmp = new ArrayList<>();
                while (!s2.empty()){
                    TreeNode node = s2.pop();
                    if (node != null){
                        tmp.add(node.val);
                        s1.push(node.right);
                        s1.push(node.left);
                    }
                }
                if (!tmp.isEmpty()){
                    result.add(tmp);
                    layer++;
                }
            }
        }

        return result;
    }

    // 52. 剑指offer：把二叉树打印成多行
    ArrayList<ArrayList<Integer> > Print2(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (pRoot == null) return result;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(pRoot);
        int star = 0, end = 1;
        while (!queue.isEmpty()){
            ArrayList<Integer> sequence = new ArrayList<>();
            for (int i = star; i <end; i++){
                TreeNode node = queue.poll();
                sequence.add(node.val);
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            result.add(sequence);
            end = queue.size();
        }
        return result;
    }

    // 53. 剑指offer：序列化二叉树
    String Serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        if (root == null) {
            sb.append("#,");
            return String.valueOf(sb);
        }
        sb.append(root.val + ",");
        sb.append(Serialize(root.left));
        sb.append(Serialize(root.right));
         return String.valueOf(sb);
    }

    int index = -1;
    TreeNode Deserialize(String str) {
        index++;
        TreeNode node = null;
        String[] cs = str.split(",");
        if (!cs[index].equals("#")){
            node = new TreeNode(Integer.parseInt(cs[index]));
            node.left = Deserialize(str);
            node.right = Deserialize(str);
        }
        return node;
    }




    public static void main(String[] args){
        FindContinuousSequence(9);
    }
}
