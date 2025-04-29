import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;


public class imageProcessor {
    public static final double[][] SOBEL_DX = {
            {-1.0, 0.0, 1.0},
            {-2.0, 0.0, 2.0},
            {-1.0, 0.0, 1.0}
    };

    public static final double[][] SOBEL_DY = {
            {-1.0, -2.0, -1.0},
            {0.0, 0.0, 0.0},
            {1.0, 2.0, 1.0}
    };

    public static BufferedImage convertToBufferedImage(double[][] data) {
        int height = data.length;
        int width = data[0].length;

        // 找到最大值和最小值
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (data[y][x] > max) max = data[y][x];
                if (data[y][x] < min) min = data[y][x];
            }
        }

        // 创建 BufferedImage
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 归一化到 0-255
                int value = (int) ((data[y][x] - min) / (max - min) * 255);
                image.setRGB(x, y, new java.awt.Color(value, value, value).getRGB());
            }
        }

        return image;
    }


    //双三次插值
    public static double[][] resize(double[][] img, double scale) {
        BufferedImage orig = toBufferedImage(img);
        BufferedImage scaledImg = new BufferedImage(
                (int)(orig.getWidth() * scale),
                (int)(orig.getHeight() * scale),
                BufferedImage.TYPE_BYTE_GRAY
        );
        AffineTransform at = new AffineTransform();
        at.scale(scale, scale);
        AffineTransformOp op = new AffineTransformOp(at, AffineTransformOp.TYPE_BICUBIC);
        op.filter(orig, scaledImg);
        return convertToGray(scaledImg);
    }

    // 图像缩放（最近邻插值）
//    public static double[][] resize(double[][] img, double scale) {
//        int originalHeight = img.length;
//        int originalWidth = img[0].length;
//        int newWidth = (int) (originalWidth * scale);
//        int newHeight = (int) (originalHeight * scale);
//
//        double[][] scaled = new double[newHeight][newWidth];
//
//        //插值算法
//        for (int y = 0; y < newHeight; y++) {
//            for (int x = 0; x < newWidth; x++) {
//                int originalX = (int) (x / scale);
//                int originalY = (int) (y / scale);
//                if (originalY >= 0 && originalY < originalHeight && originalX >= 0 && originalX < originalWidth) {
//                    scaled[y][x] = img[originalY][originalX];
//                }
//            }
//        }
//
//        return scaled;
//    }

    // 转换为灰度图
    public static double[][] convertToGray(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        double[][] gray = new double[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = new Color(img.getRGB(x, y));
                gray[y][x] = 0.2989 * color.getRed() + 0.5870 * color.getGreen() + 0.1140 * color.getBlue();
            }
        }
        return gray;
    }

    // 卷积操作
    public static double[][] convolve(double[][] input, double[][] kernel) {
        int height = input.length;
        int width = input[0].length;
        int kSize = kernel.length;
        int pad = kSize / 2;

        double[][] output = new double[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0;

                for (int ky = -pad; ky <= pad; ky++) {
                    for (int kx = -pad; kx <= pad; kx++) {
                        int cy = y + ky;
                        int cx = x + kx;


//                        // 边界处理：使用零填充
//                        if (cy >= 0 && cy < height && cx >= 0 && cx < width) {
//                            sum += input[cy][cx] * kernel[ky + pad][kx + pad];
//                        } else {
//                            sum += 0;  // 零填充

//                        // 边界处理：使用边缘填充
//                        if (cy < 0) cy = 0;
//                        if (cy >= height) cy = height - 1;
//                        if (cx < 0) cx = 0;
//                        if (cx >= width) cx = width - 1;
//                        sum += input[cy][cx] * kernel[ky + pad][kx + pad];

                        // 反射填充
                        cy = reflect(cy, height);
                        cx = reflect(cx, width);

                        sum += input[cy][cx] * kernel[ky + pad][kx + pad];

                    }
                }
                output[y][x] = sum;
            }
        }
        return output;
    }

    // 计算反射填充的坐标
//    private static int reflect(int coord, int size) {
//        while (true) {
//            if (coord < 0) {
//                coord = -coord - 1;
//            } else if (coord >= size) {
//                coord = 2 * size - coord - 1;
//            } else {
//                break;
//            }
//        }
//        return coord;
//    }
    private static int reflect(int coord, int size) {
        if (coord < 0) {
            // 反射到左边
            return -coord;
        } else if (coord >= size) {
            // 反射到右边
            return 2 * size - coord - 1;
        }
        return coord;
    }

    // 计算梯度幅值
    private static double[][] calculateMagnitude(double[][] dx, double[][] dy) {
        int height = dx.length;
        int width = dx[0].length;
        double[][] mag = new double[height][width];

        double max = Double.MIN_VALUE;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                mag[y][x] = Math.sqrt(dx[y][x] * dx[y][x] + dy[y][x] * dy[y][x]);
                if (mag[y][x] > max) max = mag[y][x];
            }
        }

        // 归一化到0-255
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                mag[y][x] = 255 * mag[y][x] / max;
            }
        }
        return mag;
    }

    // 计算梯度方向
    private static double[][] calculateOrientation(double[][] dx, double[][] dy) {
        int height = dx.length;
        int width = dx[0].length;
        double[][] orient = new double[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double angle = Math.atan2(dy[y][x], dx[y][x]);
                orient[y][x] = (Math.toDegrees(angle) + 180)*255/360; // 转换为0-360度
            }
        }
        return orient;
    }

    // 将double[][]转换为BufferedImage
    public static BufferedImage toBufferedImage(double[][] data) {
        int height = data.length;
        int width = data[0].length;
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

//        double max = Double.MIN_VALUE;
//        for (int y = 0; y < height; y++) {
//            for (int x = 0; x < width; x++) {
//                if (data[y][x] > max) max = data[y][x];
//            }
//        }
        Maximum Max=findMaximum(data);

        if (Max.max == 0) Max.max = 1;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int value = (int) (255 * data[y][x] / Max.max);
                img.setRGB(x, y, new Color(value, value, value).getRGB());
            }
        }

        return img;
    }

    // 显示图片
    public static void displayImage(double[][] data, String title, double scale, int originalWidth, int originalHeight) {
        BufferedImage img = toBufferedImage(data);
        JFrame frame = new JFrame(title);
        ImageLabel label = new ImageLabel(img, scale, originalWidth, originalHeight);
        label.setIcon(new ImageIcon(img));
        frame.getContentPane().add(new JLabel(new ImageIcon(img)));
        frame.pack();
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        //frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }

    public static class Maximum {
        public double max;
        public int x;
        public int y;

        public Maximum(double max, int x, int y) {
            this.max = max;
            this.x = x;
            this.y = y;
        }
    }

    //用于验证，找寻最大值以及坐标
    public static Maximum findMaximum(double[][] array) {
        Maximum max = new Maximum(array[0][0], 0, 0);

        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                if (array[i][j] > max.max) {
                    max.max = array[i][j];
                    max.x = j;
                    max.y = i;
                }
            }
        }
        return max;
    }

    public static double[][] computeFG(double[][] G){
            Maximum Max=findMaximum(G);
            double G_max = Max.max;
            int height = G.length;
            int width = G[0].length;

            double[][] f=new double[height][width];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    f[y][x]=(G_max-G[y][x])/G_max;
                }
            }
            return f;
    }

    public static double[][] costFunction(double[][] G){
        int height = G.length;
        int width = G[0].length;

        double[][] C=new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                C[y][x]=(1)/(1+G[y][x]);
            }
        }
        return C;
    }

    public static void imgSave(double[][] img,String s) throws IOException {//保存图片，文本
        // 将 double[][] 数据转换为 BufferedImage
        BufferedImage Image = convertToBufferedImage(img);
        // 保存图像
        File output = new File(s +".jpg");
        ImageIO.write(Image, "jpg", output);
        System.out.println("图像已保存");
    }

    public static class ImageLabel extends JLabel implements MouseListener {

        private BufferedImage image;
        private double scale; // 缩放比例
        private int originalWidth; // 原图宽度
        private int originalHeight; // 原图高度

        public ImageLabel(BufferedImage img, double scale, int originalWidth, int originalHeight) {
            this.image = img;
            this.scale = scale;
            this.originalWidth = originalWidth;
            this.originalHeight = originalHeight;
            this.addMouseListener(this);
        }

        @Override
        public void mouseClicked(MouseEvent e) {
            int x = e.getX();
            int y = e.getY();

            if (x >= 0 && x < image.getWidth() && y >= 0 && y < image.getHeight()) {
                // 获取缩放后的图片中的坐标
                System.out.println("Clicked on scaled image at (" + x + ", " + y + ")");

                // 计算原图中的大致坐标（简单按比例缩放）
                int originalX = (int) (x / scale);
                int originalY = (int) (y / scale);

                // 确保坐标在原图范围内
                originalX = Math.max(0, Math.min(originalX, originalWidth - 1));
                originalY = Math.max(0, Math.min(originalY, originalHeight - 1));

                System.out.println("Approximate original image coordinates: (" + originalX + ", " + originalY + ")");

                // 获取并打印 RGB 值
                int rgb = image.getRGB(x, y);
                Color color = new Color(rgb);
                System.out.println("RGB: " + rgb + ", Color: " + color);
                System.out.println("--------------------------------------------------");
            }
        }

        @Override
        public void mousePressed(MouseEvent e) {}

        @Override
        public void mouseReleased(MouseEvent e) {}

        @Override
        public void mouseEntered(MouseEvent e) {}

        @Override
        public void mouseExited(MouseEvent e) {}
    }

    public static void main(String[] args) {
        try {
            // 读取原始图像
            BufferedImage origImg = ImageIO.read(new File("D:\\Program\\PfJava\\project\\code\\img.jpg"));

            // 转换为灰度图
            double[][] grayData = convertToGray(origImg);
            int originalWidth = origImg.getWidth();
            int originalHeight = origImg.getHeight();

            //缩放
            double scale = 0.5;
            grayData=resize(grayData,scale);

            // 执行Sobel卷积
            double[][] dx = convolve(grayData, SOBEL_DX);
            double[][] dy = convolve(grayData, SOBEL_DY);

            // 计算幅值和方向
            double[][] magnitude = calculateMagnitude(dx, dy);
            double[][] orientation = calculateOrientation(dx, dy);

            double[][] f_G = computeFG(magnitude);
            double[][] cost = costFunction(magnitude);

            // 找到最大值及其位置
            Maximum magMax = findMaximum(magnitude);
            Maximum oriMax = findMaximum(orientation);
            Maximum fGMax = findMaximum(f_G);
            Maximum costMax = findMaximum(cost);

            System.out.println("Magnitude Max: " + magMax.max + " at (" + magMax.x + ", " + magMax.y + ")");
            System.out.println("Orientation Max: " + oriMax.max + " at (" + oriMax.x + ", " + oriMax.y + ")");

            System.out.println("f_G Max: " + fGMax.max + " at (" + fGMax.x + ", " + fGMax.y + ")");
            System.out.println("cost Max: " + costMax.max + " at (" + costMax.x + ", " + costMax.y + ")");

            // 显示结果
            displayImage(grayData, "Grayscale Image", scale, originalWidth, originalHeight);
//            displayImage(magnitude, "Gradient Magnitude", scale, originalWidth, originalHeight);
//            displayImage(orientation, "Gradient Orientation", scale, originalWidth, originalHeight);
//            displayImage(f_G, "f_G", scale, originalWidth, originalHeight);
//            displayImage(cost, "cost", scale, originalWidth, originalHeight);

//            imgSave(grayData,"grayData");
//            imgSave(magnitude,"magnitude");
//            imgSave(cost,"cost");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}