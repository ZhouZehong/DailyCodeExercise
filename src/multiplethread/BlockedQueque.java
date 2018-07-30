package multiplethread;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class BlockedQueque {
    public static void main(String[] args){
        BlockingQueue blockingQueue = new ArrayBlockingQueue(3, false);

        Thread proThread = new Thread(new Producer1(blockingQueue));
        Thread conThread = new Thread(new Consumer1(blockingQueue));

        proThread.start();
        conThread.start();
    }

}

class Producer1 implements Runnable{
    private BlockingQueue queue;
    public Producer1(BlockingQueue blockingQueue){
        queue = blockingQueue;
    }
    public void run(){
        for (int i = 0; i < 15; i++){
            try {
                System.out.println("Product: " + i);
                queue.put(i);
            }catch (InterruptedException e){
                e.printStackTrace();
            }
        }
    }
}

class Consumer1 implements Runnable{
    private BlockingQueue queue;
    public Consumer1(BlockingQueue blockingQueue){
        queue = blockingQueue;
    }
    @Override
    public void run() {
        while (true) {
            try {
                System.out.println("Consumed: " + queue.take());
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
