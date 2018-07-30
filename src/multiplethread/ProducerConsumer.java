package multiplethread;

import java.util.ArrayList;
import java.util.Random;

// 阻塞队列
class Queue{
    int maxLength;
    ArrayList<Integer> list;

    public Queue(ArrayList<Integer> a, int m){
        list = a;
        maxLength = m;
    }

    public synchronized void product(int number){
        while (list.size() == maxLength){
            try {
                this.wait();
            }catch (InterruptedException e){
                e.printStackTrace();
            }
        }
        list.add(number);
        notifyAll();
    }

    public synchronized int consume(){
        while (list.size() == 0){
            try {
                this.wait();
            }catch (InterruptedException e){
                e.printStackTrace();
            }
        }
        notifyAll();
        return list.remove(0);
    }
}

// 生产者
class Producer implements Runnable{
    private Queue queue;
    public Producer(Queue q){
        queue = q;
    }
    @Override
    public void run() {
        for (int i = 0; i < 15; i++){
            System.out.println("Product: " + i);
            queue.product(i);
            try {
                Thread.sleep(500);
            }catch (InterruptedException e){
                e.printStackTrace();
            }
        }
    }
}

// 消费者
class Consumer implements Runnable{
    private Queue queue;
    public Consumer(Queue q){
        queue = q;
    }
    @Override
    public void run() {
        while (true){
            System.out.println("Consumed: " + queue.consume());
            try {
                Thread.sleep(1000);
            }catch (InterruptedException e){
                e.printStackTrace();
            }
        }
    }
}

public class ProducerConsumer {
    public static void main(String[] args){
        ArrayList<Integer> arrayList = new ArrayList<>();
        int maxLength = 3;
        Queue queue = new Queue(arrayList, maxLength);

        Thread proThread = new Thread(new Producer(queue));
        Thread conThread = new Thread(new Consumer(queue));

        proThread.start();
        conThread.start();
    }
}
