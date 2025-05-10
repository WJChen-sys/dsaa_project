import java.awt.*;
import java.awt.event.*;
import java.awt.geom.Area;
import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;
import javax.swing.*;
import javax.imageio.ImageIO;
import javax.swing.filechooser.FileNameExtensionFilter;

// 整合项目 智能剪刀类 //

//主功能 和 GUI实现 （包括cursor snap和path cooling）
public class IntelligentScissors extends JFrame {

    private Point seedPoint = null; //种子点 点击的起始点
    private List<List<Point>> allPaths = new ArrayList<>();  //存储所有路径
    private double[][] gradients; //图像梯度
    private double[][] costMatrix; //代价 用于Dijkstra算法
    private Map<Point, Point> parentMap; //每个点的父节点，用于重建路径
    private List<Point> currentPath = new ArrayList<>(); //当前路径
    private double scaleRatioX; //缩放比例X
    private double scaleRatioY; //缩放比例Y
    //数据存储 以及算法实现需要的变量

    private BufferedImage originalImage; //原始图像
    private BufferedImage displayImage; //显示图像
    private BufferedImage scaledImage; // 缩放后的图片
    private JPanel imagePanel; //显示图像面板
    private boolean pathCompleted = false; //当前路径完成
    private Color pathColor = Color.GREEN; //路径颜色 默认绿色
    private boolean isCoolingEnabled = true; //控制path cooling开关 默认打开
    //GUI交互变量

    private Map<Point, Long> pointStability = new HashMap<>(); //哈希表记录点 用于path cooling
    private static final long COOLING_THRESHOLD_MS = 1; //1ms冷却阈值（可调整） 用于path cooling
    private static final int MIN_COOLING_LENGTH=7;      //最小冷却长度（可调整） 用于path cooling
    private Map<Point, Integer> coalescenceCount = new HashMap<>(); //记录路径聚合次数 用于path cooling
    private static final int COOLING_COUNT_THRESHOLD=4; //阈值（可调整） 冷却触发的最小聚合次数 用于path cooling
    //Path Cooling变量与阈值 调参
    //以上为全局变量//

    //构造类//
    public IntelligentScissors() {
        setTitle("Intelligent Scissors"); //窗口标题
        setSize(800, 600); //窗口大小
        setLocationRelativeTo(null); //窗口居中
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); //关闭操作

        //工具栏（打开图像、清除路径、更改颜色的按钮、路径冷却开关和导出切割图片）
        JToolBar toolBar = new JToolBar();
        JButton openButton = new JButton("Open");
        openButton.addActionListener(e -> openImage()); //图像事件监听
        JButton clearButton = new JButton("Clear");
        clearButton.addActionListener(e -> clearAllPaths()); //清除所有路径监听
        JToggleButton coolingToggleButton = new JToggleButton("Path Cooling: ON");
        coolingToggleButton.addActionListener(e -> {
            isCoolingEnabled = !isCoolingEnabled;
            if (isCoolingEnabled) {
                coolingToggleButton.setText("Path Cooling: ON");
            } else {
                coolingToggleButton.setText("Path Cooling: OFF");
            }
        });//路径冷却开关监听
        JButton exportButton = new JButton("Export Selection");
        exportButton.addActionListener(e -> exportSelection());
        //导出所选区域的切割图片监听

        //依次在工具栏添加按钮
        toolBar.add(openButton);
        toolBar.add(clearButton);
        toolBar.add(exportButton);
        toolBar.add(coolingToggleButton);

        //图像显示面板 paintComponent绘制图像与路径
        imagePanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                if (displayImage != null) {
                    g.drawImage(displayImage, 0, 0, this); // 绘制原始图像

                    // 绘制已完成路径（红色）
                    Graphics2D g2dAll = (Graphics2D) g.create();
                    g2dAll.setColor(Color.RED);
                    g2dAll.setStroke(new BasicStroke(2));
                    for (List<Point> path : allPaths) {
                        for (int i = 0; i < path.size() - 1; i++) {
                            Point p1 = path.get(i);
                            Point p2 = path.get(i + 1);

                            // 将缩放后的图片坐标转换为原始图片的坐标
                            int originalX1 = (int) (p1.x * scaleRatioX);
                            int originalY1 = (int) (p1.y * scaleRatioY);
                            int originalX2 = (int) (p2.x * scaleRatioX);
                            int originalY2 = (int) (p2.y * scaleRatioY);

                            g2dAll.drawLine(originalX1, originalY1, originalX2, originalY2);
                        }
                    }
                    g2dAll.dispose();

                    // 绘制当前路径（绿色）
                    if (!currentPath.isEmpty()) {
                        Graphics2D g2dCurrent = (Graphics2D) g.create();
                        g2dCurrent.setColor(pathColor);
                        g2dCurrent.setStroke(new BasicStroke(2));
                        for (int i = 0; i < currentPath.size() - 1; i++) {
                            Point p1 = currentPath.get(i);
                            Point p2 = currentPath.get(i + 1);

                            // 将缩放后的图片坐标转换为原始图片的坐标
                            int originalX1 = (int) (p1.x * scaleRatioX);
                            int originalY1 = (int) (p1.y * scaleRatioY);
                            int originalX2 = (int) (p2.x * scaleRatioX);
                            int originalY2 = (int) (p2.y * scaleRatioY);

                            g2dCurrent.drawLine(originalX1, originalY1, originalX2, originalY2);
                        }
                        g2dCurrent.dispose();
                    }

                    // 绘制 seed point（蓝色点）
                    if (seedPoint != null && !pathCompleted) {
                        Graphics2D g2dSeed = (Graphics2D) g.create();
                        g2dSeed.setColor(Color.BLUE);

                        // 将缩放后的图片坐标转换为原始图片的坐标
                        int originalX = (int) (seedPoint.x * scaleRatioX);
                        int originalY = (int) (seedPoint.y * scaleRatioY);

                        g2dSeed.fillOval(originalX - 4, originalY - 4, 8, 8);
                        g2dSeed.dispose();
                    }
                }
            }
        };

        //鼠标点击监听 点击选择起点和完成路径
        imagePanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (costMatrix == null) return; //无计算成本矩阵

                //在所有路径中加入未完成的路径
                if (!pathCompleted && !currentPath.isEmpty()) {
                    allPaths.add(new ArrayList<>(currentPath));
                    seedPoint = currentPath.get(currentPath.size() - 1); //路径的终点作为下一个seed point
                    currentPath.clear();
                    pathCompleted = true;//更新状态
                }

                //第一次点击or路径已完成
                if (seedPoint == null || pathCompleted) {
                    Point clickedPoint = e.getPoint();
                    int scaledX = (int) (clickedPoint.x / scaleRatioX);
                    int scaledY = (int) (clickedPoint.y / scaleRatioY);

                    Point snapPoint = findNearestHighGradientPoint(scaledX, scaledY);
                    seedPoint = snapPoint;
                    parentMap = Dijkstra.calculatePaths(seedPoint.x, seedPoint.y, costMatrix);
                    currentPath.clear();
                    currentPath.add(new Point(seedPoint.x, seedPoint.y));
                    pathCompleted = false;
                } else {
                    //用当前seed point计算路径
                    parentMap = Dijkstra.calculatePaths(seedPoint.x, seedPoint.y, costMatrix);
                    currentPath.clear();
                    currentPath.add(new Point(seedPoint.x, seedPoint.y));//状态更新
                }

                repaint(); //绘制
            }
        });

        //鼠标移动监听 鼠标移动时实时更新路径
        imagePanel.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                if (seedPoint != null && parentMap != null && !pathCompleted) {
                    Point currentMouse = e.getPoint();
                    int scaledX = (int) (currentMouse.x / scaleRatioX);
                    int scaledY = (int) (currentMouse.y / scaleRatioY);

                    Point edgePoint = findNearestHighGradientPoint(scaledX, scaledY);
                    currentPath = reconstructPath(edgePoint);

                    //更新path cooling参数
                    long timestamp = System.currentTimeMillis();
                    for (Point p : currentPath) {
                        //更新时间戳
                        pointStability.put(p, timestamp);

                        //更新聚合次数（每个点经过一次路径绘制+1）
                        coalescenceCount.put(p, coalescenceCount.getOrDefault(p, 0) + 1);
                    }

                    checkAndApplyCooling();//path cooling检查
                    repaint();//绘制
                }
            }
        });

        //GUI布局 添加工具栏和显示面板
        getContentPane().setLayout(new BorderLayout());
        getContentPane().add(toolBar, BorderLayout.NORTH);
        getContentPane().add(new JScrollPane(imagePanel), BorderLayout.CENTER);
    }

    //寻找最近高梯度点
    private Point findNearestHighGradientPoint(int x, int y) {
        int searchRadius = 7; //搜索半径 按论文设置为7
        int width = gradients.length;
        int height = gradients[0].length;
        Point bestPoint = new Point(x, y);
        double maxGradient = gradients[x][y]; //初始化 当前gradient

        for (int dx = -searchRadius; dx <= searchRadius; dx++) {
            for (int dy = -searchRadius; dy <= searchRadius; dy++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) { //图片范围限制
                    if (gradients[nx][ny] > maxGradient) {
                        maxGradient = gradients[nx][ny];
                        bestPoint = new Point(nx, ny);//找点
                    }
                }
            }
        }
        return bestPoint;
    }

    //打开文件初始化 GUI
    private void openImage() {
        JFileChooser chooser = new JFileChooser(); //文件选择器
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                originalImage = ImageIO.read(chooser.getSelectedFile()); //读取图像文件
                if (originalImage == null) {
                    JOptionPane.showMessageDialog(this, "Unsupported image format"); //不支持的格式
                    return;
                }
                //图像缩放逻辑
                int maxWidth = 800; //最大宽度
                int maxHeight = 600; //最大高度
                int imageWidth = originalImage.getWidth();
                int imageHeight = originalImage.getHeight();

                //计算缩放比例
                double ratioX = (double) maxWidth / imageWidth;
                double ratioY = (double) maxHeight / imageHeight;
                double scaleRatio = Math.min(ratioX, ratioY);

                //缩放图片
                scaledImage = new BufferedImage(
                        (int) (imageWidth * scaleRatio),
                        (int) (imageHeight * scaleRatio),
                        BufferedImage.TYPE_INT_RGB
                );
                Graphics2D g2d = scaledImage.createGraphics();
                g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                g2d.drawImage(originalImage, 0, 0, scaledImage.getWidth(), scaledImage.getHeight(), null);
                g2d.dispose();

                scaleRatioX = (double) originalImage.getWidth() / scaledImage.getWidth();
                scaleRatioY = (double) originalImage.getHeight() / scaledImage.getHeight();

                displayImage = new BufferedImage(
                        originalImage.getWidth(),
                        originalImage.getHeight(),
                        BufferedImage.TYPE_INT_RGB
                );
                displayImage.getGraphics().drawImage(originalImage, 0, 0, null);

                //计算梯度和成本矩阵（在缩放后的图片上进行计算）
                gradients = ImageProcessor.computeGradients(scaledImage);
                costMatrix = ImageProcessor.computeCostMatrix(gradients);

                //面板尺寸
                imagePanel.setPreferredSize(new Dimension(originalImage.getWidth(), originalImage.getHeight()));
                imagePanel.revalidate(); //重新布局
                pack(); //自动调整窗口大小
                setLocationRelativeTo(null); //窗口居中

                seedPoint = null; //重置种子点
                parentMap = null; //重置路径父节点映射
                allPaths.clear(); //清除所有路径
                currentPath.clear(); //清除当前路径
                pathCompleted = false; //路径状态未完成
                repaint(); //绘制
            } catch (IOException ex) {
                JOptionPane.showMessageDialog(this, "Error loading image"); //加载报错
            }
        }
    }

    //Dijkstra寻路的终点 回溯算法
    //重建路径 输入终点 返回点列表（路径）
    private List<Point> reconstructPath(Point end) {
        List<Point> path = new ArrayList<>(); //路径列表初始化
        if (end.x < 0 || end.x >= costMatrix.length || end.y < 0 || end.y >= costMatrix[0].length) {
            return path; //边界限制 直接返回
        }
        Point current = end;
        //从终点出发 父节点映射追溯起点
        int maxSteps = costMatrix.length * costMatrix[0].length; //最大循环次数
        int steps = 0; //初始化循环索引
        while (current != null && parentMap.containsKey(current) && steps < maxSteps) {
            path.add(current);
            current = parentMap.get(current);
            steps++; //循环次数+1
        }
        Collections.reverse(path); //路径反转 起点-->终点
        return path;
    }

    //清除所有路径
    private void clearAllPaths() {
        allPaths.clear();
        seedPoint = null;
        parentMap = null;
        currentPath.clear();
        pathCompleted = false;
        //所有状态清楚

        pointStability.clear();
        coalescenceCount.clear();
        //path cooling清除

        repaint();
    }

    //导出切割部分
    private void exportSelection() {
        if (allPaths.isEmpty()) {
            JOptionPane.showMessageDialog(this, "No paths to export"); //路径为空
            return;
        }

        //合并路径->连续路径
        List<Point> mergedPath = new ArrayList<>();
        for (List<Point> path : allPaths) {
            mergedPath.addAll(path);
        }
        if (mergedPath.size() < 3) {
            JOptionPane.showMessageDialog(this, "Path too short to form a region");
            //路径太短 点数<3
            return;
        }

        //路径闭合
        Path2D.Double polygon = new Path2D.Double();
        //转换到原始图像坐标
        List<Point> originalPoints = new ArrayList<>();
        for (Point p : mergedPath) {
            int originalX = (int) (p.x * scaleRatioX);
            int originalY = (int) (p.y * scaleRatioY);
            originalPoints.add(new Point(originalX, originalY));
        }

        polygon.moveTo(originalPoints.get(0).x, originalPoints.get(0).y);
        for (int i = 1; i < originalPoints.size(); i++) {
            polygon.lineTo(originalPoints.get(i).x, originalPoints.get(i).y);
        }
        polygon.closePath();
        Area totalArea = new Area(polygon);

        //透明通道图像初始化
        BufferedImage result = new BufferedImage(
                originalImage.getWidth(),
                originalImage.getHeight(),
                BufferedImage.TYPE_INT_ARGB
        );

        Graphics2D g2d = result.createGraphics();
        g2d.setComposite(AlphaComposite.Clear);
        g2d.fillRect(0, 0, result.getWidth(), result.getHeight());
        g2d.setComposite(AlphaComposite.Src);

        //裁剪区域绘制
        g2d.setClip(totalArea);
        g2d.drawImage(originalImage, 0, 0, null);
        g2d.dispose();

        //保存文件
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

    //Path Cooling检查
    private void checkAndApplyCooling() {
        if (!isCoolingEnabled) return; //冷却禁用
        if (currentPath.isEmpty()) return;//路径为空

        long currentTime = System.currentTimeMillis();
        int coolingIndex = -1;

        //从后往前寻找第一个满足时间和次数条件的点
        for (int i = currentPath.size() - 1; i >= 0; i--) {
            Point p = currentPath.get(i);
            Long firstSeen = pointStability.get(p);
            Integer count = coalescenceCount.getOrDefault(p, 0);

            if (firstSeen != null &&
                    (currentTime - firstSeen) >= COOLING_THRESHOLD_MS && count >= COOLING_COUNT_THRESHOLD) {
                coolingIndex = i;
                break;
            }
        }

        if (coolingIndex >= MIN_COOLING_LENGTH) {
            //提取已冷却的路径
            List<Point> cooledSegment = new ArrayList<>(currentPath.subList(0, coolingIndex + 1));
            allPaths.add(cooledSegment);

            //更新种子点 重置路径
            seedPoint = new Point(cooledSegment.get(cooledSegment.size() - 1));
            parentMap = Dijkstra.calculatePaths(seedPoint.x, seedPoint.y, costMatrix);
            currentPath = new ArrayList<>(currentPath.subList(coolingIndex, currentPath.size()));
            currentPath.add(0, seedPoint);

            //清空记录
            pointStability.clear();
            coalescenceCount.clear(); //次数统计
            repaint();
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new IntelligentScissors().setVisible(true)); //启动程序 GUI
    }
}

// 统一整合 //
// 内部类 //

//图片处理类
class ImageProcessor {
    //灰度转化 梯度计算（输入BufferedImage类型图片 输出梯度2D-double数组）
    public static double[][] computeGradients(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[][] gradient = new double[width][height];

        //转换为灰度图
        double[][] grayImage = new double[width][height];
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                Color color = new Color(image.getRGB(x, y));
                grayImage[x][y] = 0.2989 * color.getRed() + 0.5870 * color.getGreen() + 0.1140 * color.getBlue();
            }
        }

        //Sobel算子定义
        int[][] sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int[][] sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        //卷积计算梯度
        for (int x = 1; x < width - 1; x++) {
            for (int y = 1; y < height - 1; y++) {
                double gx = 0, gy = 0;
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        gx += grayImage[x + i][y + j] * sobelX[i + 1][j + 1];
                        gy += grayImage[x + i][y + j] * sobelY[i + 1][j + 1];
                    }
                }
                gradient[x][y] = Math.sqrt(gx * gx + gy * gy);
            }
        }

        //处理边缘像素 默认为0填充 尽量保持小梯度 不倾向识别为边缘
//        for (int x = 0; x < width; x++) {
//            for (int y = 0; y < height; y++) {
//                if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
//                    gradient[x][y] = 0; // 或者根据实际情况处理边缘像素
//                }
//            }
//        }
        return gradient;
    }

    //cost计算
    public static double[][] computeCostMatrix(double[][] gradients) {

        double[][] cost = new double[gradients.length][gradients[0].length];
        for (int x = 0; x < gradients.length; x++) {
            for (int y = 0; y < gradients[0].length; y++) {
                //每个像素的cost 梯度越大成本越低 (分母+1)防止除以0报错
                cost[x][y] = 1.0 / (1.0 + gradients[x][y]);
            }
        }
        return cost;
    }
}


// Dijkstra算法实现 //
class Dijkstra {

    //Dijkstra最短路径算法 （分别输入开始点的X和Y坐标 以及 cost的2D-double数组，输出点与点的hashmap） <--主功能的父节点映射
    static Map<Point, Point> calculatePaths(int startX, int startY, double[][] cost) {
        int width = cost.length; //宽度
        int height = cost[0].length; //高度
        double[][] dist = new double[width][height]; //距离矩阵初始化
        //算法实现
        for (double[] row : dist) Arrays.fill(row, Double.POSITIVE_INFINITY); //初始化距离为无穷大
        dist[startX][startY] = 0; //初始化起始点距离为0

        PriorityQueue<Node> queue = new PriorityQueue<>(); //优先队列初始化
        queue.add(new Node(startX, startY, 0)); //把起始点添加到队列

        Map<Point, Point> parentMap = new HashMap<>(); //创建输出map <--父节点映射
        int[][] dirs = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}}; //8个方向
        double[] distances = {Math.sqrt(2), 1, Math.sqrt(2), 1, 1, Math.sqrt(2), 1, Math.sqrt(2)};
        //8个方向的距离 其中，对角线为根号2

        while (!queue.isEmpty()) { //队列不为空 进入循环
            Node current = queue.poll(); //从队列获得当前节点
            if (current.dist > dist[current.x][current.y]) continue; //当前距离大于已知距离

            int idx=0;//循环索引
            for (int[] dir : dirs) { //遍历所有方向
                int nx = current.x + dir[0]; //新节点X坐标
                int ny = current.y + dir[1]; //新节点Y坐标
                double distance = distances[idx];//新节点的距离
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) { //新节点是否在图片范围内
                    double newDist = current.dist + cost[nx][ny]* distance; //更新距离计算
                    if (newDist < dist[nx][ny]) { //新距离更小
                        dist[nx][ny] = newDist; //找到更小的距离 贪心算法
                        parentMap.put(new Point(nx, ny), new Point(current.x, current.y)); //更新map <--父节点映射
                        queue.add(new Node(nx, ny, newDist)); //把新节点添加到队列
                    }
                }
                idx++;//循环索引更新 idx+1
            }
        }
        return parentMap; //返回map <--父节点映射
    }

    //节点类 用于优先队列
    static class Node implements Comparable<Node> {
        int x, y; //节点坐标
        double dist; //节点距离

        //构造函数
        Node(int x, int y, double dist) {
            this.x = x;
            this.y = y;
            this.dist = dist;
        }

        //比较方法
        @Override
        public int compareTo(Node other) {
            return Double.compare(this.dist, other.dist); //按距离排序 从小到大
        }
    }
}