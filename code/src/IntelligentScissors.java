import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.Area;
import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;
import javax.imageio.ImageIO;
import javax.swing.filechooser.FileNameExtensionFilter;

public class IntelligentScissors extends JFrame {
    private BufferedImage originalImage; // 原始图像
    private BufferedImage displayImage; // 显示图像
    private Point seedPoint = null; // 种子点，用户点击的起始点
    private List<List<Point>> allPaths = new ArrayList<>();  // 存储所有已完成的路径
    private double[][] costMatrix; // 成本矩阵，用于Dijkstra算法
    private Map<Point, Point> parentMap; // 记录每个点的父节点，用于重建路径
    private List<Point> currentPath = new ArrayList<>(); // 当前正在绘制的路径
    private JPanel imagePanel; // 显示图像的面板
    private boolean pathCompleted = false; // 标志当前路径是否已完成
    private Color pathColor = Color.GREEN; // 路径的颜色，默认为绿色
    private double[][] gradients; // 新增：存储图像梯度数据
    private Map<Point, Long> pointStability = new HashMap<>();
    private static final long COOLING_THRESHOLD_MS = 1; // 2秒冷却阈值
    private static final int MIN_COOLING_LENGTH = 7;      // 最小冷却段长度
    private Map<Point, Integer> coalescenceCount = new HashMap<>(); // 路径聚合次数计数器
    private static final int COOLING_COUNT_THRESHOLD = 4; // 冷却触发的最小聚合次数



    public IntelligentScissors() {
        setTitle("Intelligent Scissors"); // 设置窗口标题
        setSize(800, 600); // 设置窗口大小
        setLocationRelativeTo(null); // 窗口居中
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 设置关闭操作

        // 创建工具栏，包含打开图像、清除路径和更改颜色的按钮
        JToolBar toolBar = new JToolBar();
        JButton openButton = new JButton("Open");
        openButton.addActionListener(e -> openImage()); // 打开图像的事件监听器
        JButton clearButton = new JButton("Clear");
        clearButton.addActionListener(e -> clearAllPaths()); // 清除所有路径的事件监听器
        JButton colorButton = new JButton("Change Color");
        colorButton.addActionListener(e -> changeColor()); // 更改路径颜色的事件监听器

        JButton exportButton = new JButton("Export Selection");
        exportButton.addActionListener(e -> exportSelection());

        // 添加按钮到工具栏
        toolBar.add(openButton);
        toolBar.add(clearButton);
        toolBar.add(colorButton);
        toolBar.add(exportButton);

        // 创建图像显示面板，并重写paintComponent方法来绘制图像和路径
        imagePanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                if (displayImage != null) {
                    g.drawImage(displayImage, 0, 0, this); // 绘制背景图像
                    // 绘制所有已完成的路径
                    Graphics2D g2d = (Graphics2D) g;
                    g2d.setColor(pathColor); // 设置路径颜色
                    g2d.setStroke(new BasicStroke(2)); // 设置线条宽度
                    for (List<Point> path : allPaths) {
                        for (int i = 0; i < path.size() - 1; i++) {
                            Point p1 = path.get(i);
                            Point p2 = path.get(i + 1);
                            g2d.drawLine(p1.x, p1.y, p2.x, p2.y); // 绘制路径线段
                        }
                    }
                    // 绘制当前正在绘制的路径
                    if (!currentPath.isEmpty()) {
                        for (int i = 0; i < currentPath.size() - 1; i++) {
                            Point p1 = currentPath.get(i);
                            Point p2 = currentPath.get(i + 1);
                            g2d.drawLine(p1.x, p1.y, p2.x, p2.y); // 绘制路径线段
                        }
                    }
                }
            }
        };

        // 添加鼠标点击事件监听器，处理用户点击选择起点和完成路径的操作
        imagePanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (costMatrix == null) return; // 如果没有计算成本矩阵，则不执行操作

                // 如果当前有未完成的路径，则将其添加到所有路径中
                if (!pathCompleted && !currentPath.isEmpty()) {
                    allPaths.add(new ArrayList<>(currentPath));
                    seedPoint = currentPath.get(currentPath.size() - 1); // 设置下一个起点为当前路径的终点
                    currentPath.clear();
                    pathCompleted = true;
                }

                // 如果是第一次点击或者路径已完成，则设置新的种子点
                if (seedPoint == null || pathCompleted) {
                    Point clickedPoint = e.getPoint();
                    Point snappedPoint = findNearestHighGradientPoint(clickedPoint.x, clickedPoint.y, costMatrix);
                    seedPoint = snappedPoint;
                    parentMap = Dijkstra.calculatePaths(seedPoint.x, seedPoint.y, costMatrix);
                    currentPath.clear();
                    currentPath.add(new Point(seedPoint.x, seedPoint.y));
                    pathCompleted = false;
                } else {
                    // 使用当前种子点计算路径
                    parentMap = Dijkstra.calculatePaths(seedPoint.x, seedPoint.y, costMatrix);
                    currentPath.clear();
                    currentPath.add(new Point(seedPoint.x, seedPoint.y));
                }

                repaint(); // 重新绘制面板
            }
        });

        // 添加鼠标移动事件监听器，处理鼠标移动时实时更新路径的操作
        imagePanel.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                if (seedPoint != null && parentMap != null && !pathCompleted) {
                    Point currentMouse = e.getPoint();
                    Point edgePoint = findNearestHighGradientPoint(currentMouse.x, currentMouse.y, costMatrix);
                    currentPath = reconstructPath(edgePoint);

                    // 更新稳定性时间戳和聚合次数
                    long timestamp = System.currentTimeMillis();
                    for (Point p : currentPath) {
                        // 更新时间戳
                        pointStability.put(p, timestamp);

                        // 更新聚合次数（每个点经过一次路径绘制+1）
                        coalescenceCount.put(p, coalescenceCount.getOrDefault(p, 0) + 1);
                    }

                    checkAndApplyCooling();
                    repaint();
                }
            }
        });

        // 设置窗口布局，并添加工具栏和图像显示面板
        getContentPane().setLayout(new BorderLayout());
        getContentPane().add(toolBar, BorderLayout.NORTH);
        getContentPane().add(new JScrollPane(imagePanel), BorderLayout.CENTER);
    }

    // 找到离指定点最近的梯度最大的点
    // 找到离指定点最近的梯度最大的点
    private Point findNearestHighGradientPoint(int x, int y, double[][] costMatrix) {
        int searchRadius = 10; // 搜索按钮
        int width = gradients.length;
        int height = gradients[0].length;
        Point bestPoint = new Point(x, y);
        double maxGradient = gradients[x][y]; // 初始设为当前点的梯度

        for (int dx = -searchRadius; dx <= searchRadius; dx++) {
            for (int dy = -searchRadius; dy <= searchRadius; dy++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) { // 确保在图片范围内
                    if (gradients[nx][ny] > maxGradient) {
                        maxGradient = gradients[nx][ny];
                        bestPoint = new Point(nx, ny);
                    }
                }
            }
        }

        return bestPoint;
    }

    // 打开图像文件并进行初始化
    private void openImage() {
        JFileChooser chooser = new JFileChooser(); // 创建文件选择器
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                originalImage = ImageIO.read(chooser.getSelectedFile()); // 读取图像文件
                if (originalImage == null) {
                    JOptionPane.showMessageDialog(this, "Unsupported image format"); // 提示不支持的图像格式
                    return;
                }
                displayImage = new BufferedImage(
                        originalImage.getWidth(),
                        originalImage.getHeight(),
                        BufferedImage.TYPE_INT_RGB
                );
                displayImage.getGraphics().drawImage(originalImage, 0, 0, null); // 将图像复制到显示缓冲区

                // 计算图像梯度并生成成本矩阵
                gradients = ImageProcessor.computeGradients(originalImage);
                costMatrix = computeCostMatrix(gradients);

                // 设置面板尺寸
                imagePanel.setPreferredSize(new Dimension(originalImage.getWidth(), originalImage.getHeight()));
                imagePanel.revalidate(); // 触发重新布局
                pack(); // 自动调整窗口大小
                setLocationRelativeTo(null); // 窗口居中

                seedPoint = null; // 重置种子点
                parentMap = null; // 重置路径映射
                allPaths.clear(); // 清除所有路径
                currentPath.clear(); // 清除当前路径
                pathCompleted = false; // 标志路径为未完成状态
                repaint(); // 重新绘制面板
            } catch (IOException ex) {
                JOptionPane.showMessageDialog(this, "Error loading image"); // 提示图像加载错误
            }
        }
    }

    // 根据终点重建路径
    private List<Point> reconstructPath(Point end) {
        List<Point> path = new ArrayList<>(); // 创建路径列表
        if (end.x < 0 || end.x >= costMatrix.length ||
                end.y < 0 || end.y >= costMatrix[0].length) {
            return path; // 边界检查
        }
        Point current = end;
        // 从终点开始，通过父节点映射回溯到起点
        int maxSteps = costMatrix.length * costMatrix[0].length; // 防止无限循环
        int steps = 0;
        while (current != null && parentMap.containsKey(current) && steps < maxSteps) {
            path.add(current);
            current = parentMap.get(current);
            steps++;
        }
        Collections.reverse(path); // 反转路径，使其从起点到终点
        return path;
    }

    // 计算成本矩阵
    private double[][] computeCostMatrix(double[][] gradients) {
        double maxG = Arrays.stream(gradients).flatMapToDouble(Arrays::stream).max().orElse(1.0); // 计算最大梯度值
        double[][] cost = new double[gradients.length][gradients[0].length];
        for (int x = 0; x < gradients.length; x++) {
            for (int y = 0; y < gradients[0].length; y++) {
                // 计算每个像素的成本值，梯度越大成本越低
                cost[x][y] = 1.0 / (1.0 + gradients[x][y]);
            }
        }
        return cost;
    }

    // 清除所有路径
    private void clearAllPaths() {
        allPaths.clear();
        seedPoint = null;
        parentMap = null;
        currentPath.clear();
        pathCompleted = false;

        // 新增：清空冷却相关数据
        pointStability.clear();
        coalescenceCount.clear();

        repaint();
    }

    // 更改路径颜色
    private void changeColor() {
        Color newColor = JColorChooser.showDialog(this, "Choose Path Color", pathColor); // 显示颜色选择器
        if (newColor != null) {
            pathColor = newColor; // 更新路径颜色
            repaint(); // 重新绘制面板
        }
    }

    private void exportSelection() {
        if (allPaths.isEmpty()) {
            JOptionPane.showMessageDialog(this, "No paths to export");
            return;
        }

        // 合并所有路径为一个连续路径
        List<Point> mergedPath = new ArrayList<>();
        for (List<Point> path : allPaths) {
            mergedPath.addAll(path);
        }
        if (mergedPath.size() < 3) {
            JOptionPane.showMessageDialog(this, "Path too short to form a region");
            return;
        }

        // 创建闭合路径
        Path2D.Double polygon = new Path2D.Double();
        polygon.moveTo(mergedPath.get(0).x, mergedPath.get(0).y);
        for (int i = 1; i < mergedPath.size(); i++) {
            polygon.lineTo(mergedPath.get(i).x, mergedPath.get(i).y);
        }
        polygon.closePath();
        Area totalArea = new Area(polygon);

        // 创建带透明通道的图像
        BufferedImage result = new BufferedImage(
                originalImage.getWidth(),
                originalImage.getHeight(),
                BufferedImage.TYPE_INT_ARGB
        );

        Graphics2D g2d = result.createGraphics();
        g2d.setComposite(AlphaComposite.Clear);
        g2d.fillRect(0, 0, result.getWidth(), result.getHeight());
        g2d.setComposite(AlphaComposite.Src);

        // 设置裁剪区域并绘制图像
        g2d.setClip(totalArea);
        g2d.drawImage(originalImage, 0, 0, null);
        g2d.dispose();

        // 保存文件
        JFileChooser chooser = new JFileChooser();
        chooser.setFileFilter(new FileNameExtensionFilter("PNG Image", "png"));
        if (chooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                ImageIO.write(result, "PNG", chooser.getSelectedFile());
                JOptionPane.showMessageDialog(this, "Export successful");
            } catch (IOException ex) {
                JOptionPane.showMessageDialog(this, "Export failed: " + ex.getMessage());
            }
        }
    }
    private void checkAndApplyCooling() {
        if (currentPath.isEmpty()) return;

        long currentTime = System.currentTimeMillis();
        int coolingIndex = -1;

        // 从后往前寻找第一个同时满足时间和次数条件的点
        for (int i = currentPath.size() - 1; i >= 0; i--) {
            Point p = currentPath.get(i);
            Long firstSeen = pointStability.get(p);
            Integer count = coalescenceCount.getOrDefault(p, 0);

            if (firstSeen != null &&
                    (currentTime - firstSeen) >= COOLING_THRESHOLD_MS &&
                    count >= COOLING_COUNT_THRESHOLD) {
                coolingIndex = i;
                break;
            }
        }

        if (coolingIndex >= MIN_COOLING_LENGTH) {
            // 提取已冷却的路径段
            List<Point> cooledSegment = new ArrayList<>(currentPath.subList(0, coolingIndex + 1));
            allPaths.add(cooledSegment);

            // 更新种子点并重置路径
            seedPoint = new Point(cooledSegment.get(cooledSegment.size() - 1));
            parentMap = Dijkstra.calculatePaths(seedPoint.x, seedPoint.y, costMatrix);
            currentPath = new ArrayList<>(currentPath.subList(coolingIndex, currentPath.size()));
            currentPath.add(0, seedPoint);

            // 清空稳定性记录
            pointStability.clear();
            coalescenceCount.clear(); // 清除次数统计
            repaint();
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new IntelligentScissors().setVisible(true)); // 启动应用程序
    }
}

class ImageProcessor {
    // 计算图像梯度
    public static double[][] computeGradients(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[][] gradients = new double[width][height];

        // 将图像转换为灰度图
        double[][] gray = new double[width][height];
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                Color color = new Color(image.getRGB(x, y));
                gray[x][y] = 0.2989 * color.getRed() + 0.5870 * color.getGreen() + 0.1140 * color.getBlue();
            }
        }

        // 使用Sobel算子计算梯度
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

        // 处理边缘像素
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
                    gradients[x][y] = 0; // 或者根据实际情况处理边缘像素
                }
            }
        }

        return gradients;
    }
}

class Dijkstra {
    // 使用Dijkstra算法计算最短路径
    static Map<Point, Point> calculatePaths(int startX, int startY, double[][] cost) {
        int width = cost.length; // 获取成本矩阵宽度
        int height = cost[0].length; // 获取成本矩阵高度
        double[][] dist = new double[width][height]; // 创建距离矩阵
        for (double[] row : dist) Arrays.fill(row, Double.POSITIVE_INFINITY); // 初始化距离为无穷大
        dist[startX][startY] = 0; // 起始点距离为0

        PriorityQueue<Node> queue = new PriorityQueue<>(); // 创建优先队列
        queue.add(new Node(startX, startY, 0)); // 将起始点添加到队列

        Map<Point, Point> parentMap = new HashMap<>(); // 创建父节点映射
        int[][] dirs = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}}; // 定义8个方向

        while (!queue.isEmpty()) { // 当队列不为空时循环
            Node current = queue.poll(); // 获取当前节点
            if (current.dist > dist[current.x][current.y]) continue; // 如果当前距离大于已知距离，则跳过

            for (int[] dir : dirs) { // 遍历所有方向
                int nx = current.x + dir[0]; // 计算新节点X坐标
                int ny = current.y + dir[1]; // 计算新节点Y坐标
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) { // 检查新节点是否在范围内
                    double newDist = current.dist + cost[nx][ny]; // 计算新距离
                    if (newDist < dist[nx][ny]) { // 如果新距离更小
                        dist[nx][ny] = newDist; // 更新距离
                        parentMap.put(new Point(nx, ny), new Point(current.x, current.y)); // 更新父节点映射
                        queue.add(new Node(nx, ny, newDist)); // 将新节点添加到队列
                    }
                }
            }
        }
        return parentMap; // 返回父节点映射
    }

    // 定义节点类，用于优先队列
    static class Node implements Comparable<Node> {
        int x, y; // 节点坐标
        double dist; // 节点距离

        Node(int x, int y, double dist) {
            this.x = x;
            this.y = y;
            this.dist = dist;
        }

        @Override
        public int compareTo(Node other) {
            return Double.compare(this.dist, other.dist); // 按距离排序
        }
    }
}