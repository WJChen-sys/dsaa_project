import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;
import javax.imageio.ImageIO;

public class IntelligentScissors extends JFrame {
    private BufferedImage originalImage;
    private BufferedImage displayImage;
    private Point seedPoint = null;
    private List<List<Point>> allPaths = new ArrayList<>();  // 用于存储所有路径
    private double[][] costMatrix;
    private Map<Point, Point> parentMap;
    private List<Point> currentPath = new ArrayList<>(); // 当前拖动的路径
    private JPanel imagePanel;
    private boolean pathCompleted = false; // 标志路径是否已完成
    private Color pathColor = Color.GREEN; // 设置路径颜色为绿色

    public IntelligentScissors() {
        setTitle("Intelligent Scissors");
        setSize(800, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // 工具栏
        JToolBar toolBar = new JToolBar();
        JButton openButton = new JButton("Open");
        openButton.addActionListener(e -> openImage());
        JButton clearButton = new JButton("Clear");
        clearButton.addActionListener(e -> clearAllPaths());
        JButton colorButton = new JButton("Change Color");
        colorButton.addActionListener(e -> changeColor());
        toolBar.add(openButton);
        toolBar.add(clearButton);
        toolBar.add(colorButton);

        // 图像显示面板
        imagePanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                if (displayImage != null) {
                    g.drawImage(displayImage, 0, 0, this);
                    // 绘制所有路径
                    Graphics2D g2d = (Graphics2D) g;
                    g2d.setColor(pathColor); // 使用设置的颜色
                    g2d.setStroke(new BasicStroke(2));
                    for (List<Point> path : allPaths) {
                        for (int i = 0; i < path.size() - 1; i++) {
                            Point p1 = path.get(i);
                            Point p2 = path.get(i + 1);
                            g2d.drawLine(p1.x, p1.y, p2.x, p2.y);
                        }
                    }
                    // 绘制当前拖动的路径
                    if (!currentPath.isEmpty()) {
                        for (int i = 0; i < currentPath.size() - 1; i++) {
                            Point p1 = currentPath.get(i);
                            Point p2 = currentPath.get(i + 1);
                            g2d.drawLine(p1.x, p1.y, p2.x, p2.y);
                        }
                    }
                }
            }
        };

        // 事件监听
        imagePanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (costMatrix == null) return;

                if (!pathCompleted && !currentPath.isEmpty()) {
                    // 如果已有未完成的路径，则将其添加到所有路径中
                    allPaths.add(new ArrayList<>(currentPath));
                    currentPath.clear();
                    pathCompleted = true;
                }

                seedPoint = e.getPoint();
                parentMap = Dijkstra.calculatePaths(seedPoint.x, seedPoint.y, costMatrix);
                currentPath.clear();
                currentPath.add(new Point(seedPoint.x, seedPoint.y));
                pathCompleted = false;
                repaint();
            }
        });

        imagePanel.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                if (seedPoint != null && parentMap != null && !pathCompleted) {
                    Point currentMouse = e.getPoint();
                    currentPath = reconstructPath(currentMouse);
                    repaint();
                }
            }
        });

        // 布局
        getContentPane().setLayout(new BorderLayout());
        getContentPane().add(toolBar, BorderLayout.NORTH);
        getContentPane().add(new JScrollPane(imagePanel), BorderLayout.CENTER);
    }

    private void openImage() {
        JFileChooser chooser = new JFileChooser();
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                originalImage = ImageIO.read(chooser.getSelectedFile());
                if (originalImage == null) {
                    JOptionPane.showMessageDialog(this, "Unsupported image format");
                    return;
                }
                displayImage = new BufferedImage(
                        originalImage.getWidth(),
                        originalImage.getHeight(),
                        BufferedImage.TYPE_INT_RGB
                );
                displayImage.getGraphics().drawImage(originalImage, 0, 0, null);

                // 计算梯度矩阵
                double[][] gradients = ImageProcessor.computeGradients(originalImage);
                costMatrix = computeCostMatrix(gradients);

                seedPoint = null;
                parentMap = null;
                allPaths.clear();
                currentPath.clear();
                pathCompleted = false;
                repaint();
            } catch (IOException ex) {
                JOptionPane.showMessageDialog(this, "Error loading image");
            }
        }
    }

    private List<Point> reconstructPath(Point end) {
        List<Point> path = new ArrayList<>();
        Point current = end;
        while (current != null && parentMap.containsKey(current)) {
            path.add(current);
            current = parentMap.get(current);
        }
        Collections.reverse(path);
        return path;
    }

    private double[][] computeCostMatrix(double[][] gradients) {
        double maxG = Arrays.stream(gradients).flatMapToDouble(Arrays::stream).max().orElse(1.0);
        double[][] cost = new double[gradients.length][gradients[0].length];
        for (int x = 0; x < gradients.length; x++) {
            for (int y = 0; y < gradients[0].length; y++) {
                cost[x][y] = 1.0 / (1.0 + gradients[x][y]);
            }
        }
        return cost;
    }

    private void clearAllPaths() {
        allPaths.clear();
        seedPoint = null;
        parentMap = null;
        currentPath.clear();
        pathCompleted = false;
        repaint();
    }

    private void changeColor() {
        Color newColor = JColorChooser.showDialog(this, "Choose Path Color", pathColor);
        if (newColor != null) {
            pathColor = newColor;
            repaint();
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new IntelligentScissors().setVisible(true));
    }
}

class ImageProcessor {
    public static double[][] computeGradients(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[][] gradients = new double[width][height];

        // 转换为灰度
        double[][] gray = new double[width][height];
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                Color color = new Color(image.getRGB(x, y));
                gray[x][y] = 0.2989 * color.getRed() + 0.5870 * color.getGreen() + 0.1140 * color.getBlue();
            }
        }

        // Sobel算子
        int[][] sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int[][] sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        for (int x = 1; x < width - 1; x++) {
            for (int y = 1; y < height - 1; y++) {
                double gx = 0, gy = 0;
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        gx += gray[x + i][y + j] * sobelX[i + 1][j + 1];
                        gy += gray[x + i][y + j] * sobelY[i + 1][j + 1];
                    }
                }
                gradients[x][y] = Math.sqrt(gx * gx + gy * gy);
            }
        }
        return gradients;
    }
}

class Dijkstra {
    static Map<Point, Point> calculatePaths(int startX, int startY, double[][] cost) {
        int width = cost.length;
        int height = cost[0].length;
        double[][] dist = new double[width][height];
        for (double[] row : dist) Arrays.fill(row, Double.POSITIVE_INFINITY);
        dist[startX][startY] = 0;

        PriorityQueue<Node> queue = new PriorityQueue<>();
        queue.add(new Node(startX, startY, 0));

        Map<Point, Point> parentMap = new HashMap<>();
        int[][] dirs = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};

        while (!queue.isEmpty()) {
            Node current = queue.poll();
            if (current.dist > dist[current.x][current.y]) continue;

            for (int[] dir : dirs) {
                int nx = current.x + dir[0];
                int ny = current.y + dir[1];
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    double newDist = current.dist + cost[nx][ny];
                    if (newDist < dist[nx][ny]) {
                        dist[nx][ny] = newDist;
                        parentMap.put(new Point(nx, ny), new Point(current.x, current.y));
                        queue.add(new Node(nx, ny, newDist));
                    }
                }
            }
        }
        return parentMap;
    }

    static class Node implements Comparable<Node> {
        int x, y;
        double dist;
        Node(int x, int y, double dist) {
            this.x = x;
            this.y = y;
            this.dist = dist;
        }
        @Override
        public int compareTo(Node other) {
            return Double.compare(this.dist, other.dist);
        }
    }
}