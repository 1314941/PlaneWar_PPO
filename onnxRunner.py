import onnxruntime as ort
import numpy as np
import cv2


class OnnxRunner:
    def __init__(self, model_path, input_width=640, input_height=640, confidence_thres=0.5, iou_thres=0.4, classes=[]):
        """
        :param model_path: ONNX 模型文件的路径
        :param input_width: 模型输入的宽度
        :param input_height: 模型输入的高度
        :param confidence_thres: 过滤检测结果的置信度阈值
        :param iou_thres: 非极大值抑制的 IOU 阈值
        :param classes: 类别名称列表
        """
        self.img_height = 0
        self.img_width = 0
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input_width = input_width
        self.input_height = input_height
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = classes

    def preprocess(self, image):
        """
        预处理输入图像

        :param image: OpenCV 读取的图像
        :return: 预处理后的图像和原始图像尺寸
        """
        self.img_height, self.img_width = image.shape[:2]  # 获取图像的高度和宽度
        image_resized = cv2.resize(image, (self.input_width, self.input_height))  # 将图像调整为指定大小
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)  # 将图像从BGR格式转换为RGB格式
        image_normalized = image_rgb.astype(np.float32) / 255.0  # 将图像像素值归一化到0-1之间
        # 将图像的通道、高度和宽度进行转置
        image_transposed = np.transpose(image_normalized, (2, 0, 1))  
        # 在图像的通道维度上增加一个维度
        image_expanded = np.expand_dims(image_transposed, axis=0)  
        # 返回扩展后的图像
        return image_expanded

    def postprocess(self, outputs):
        """
        后处理模型输出

        :param outputs: 模型输出
        :return: 过滤后的检测框
        """
        outputs = np.squeeze(outputs[0])
        # print(outputs.shape)

        # 获取输出数组的行数
        rows = outputs.shape[0]

        # 存储检测到的边界框、得分和类别ID的列表
        detections = []

        # 计算边界框坐标的缩放因子
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # 遍历输出数组中的每一行
        for i in range(rows):
            # print(int(outputs[i][0]), int(outputs[i][1]), int(outputs[i][2]), int(outputs[i][3]), round(float(outputs[i][4]), 4), int(outputs[i][5]))

            # print(round(float(outputs[i][4].item()), 4))

            # 从当前行中提取类别得分
            classes_scores = round(float(outputs[i][4]), 4)

            # 如果最大得分高于置信度阈值
            if classes_scores >= self.confidence_thres:
                # 获取得分最高的类别ID
                class_id = int(outputs[i][5])

                # 从当前行中提取边界框坐标
                x1, y1, x2, y2 = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 计算边界框的缩放坐标
                left = int(x1 * x_factor)
                top = int(y1 * y_factor)
                width = int(x2 * x_factor)
                height = int(y2 * y_factor)

                # 添加检测信息到列表中
                detections.append({
                    "class_id": class_id,
                    "class_name": self.classes[class_id],
                    "score": classes_scores,
                    "box": [left, top, width, height]
                })

        # 应用非极大值抑制过滤重叠的边界框
        # indices = cv2.dnn.NMSBoxes(
        #     [det["box"] for det in detections],
        #     [det["score"] for det in detections],
        #     self.confidence_thres,
        #     self.iou_thres
        # )
        #
        # #
        # # # 根据非极大值抑制后的索引过滤检测结果
        # final_detections = [detections[i] for i in indices]

        return detections

    def run(self, image):
        """
        运行模型推理

        :param image: OpenCV 读取的图像
        :return: 过滤后的检测框
        """
# 对输入图像进行预处理
        input_data = self.preprocess(image)
# 获取模型输入的名称
        input_name = self.session.get_inputs()[0].name
# 运行模型，获取输出
        outputs = self.session.run(None, {input_name: input_data})
# 对输出进行后处理
        return self.postprocess(outputs)

    def get_max_label(self, image):
        # 运行推理
        detections = self.run(image)

        max_score = 0.0
        result = None
        for det in detections:
            # 获取每个检测结果的分数
            score = float(det["score"])
            # 如果分数大于当前最大分数，则更新最大分数和结果
            if score > max_score:
                max_score = score
                result = det["class_name"]
        return result

    def draw_detections(self, image, detections):
        """
        在图像上绘制检测结果

        :param image: OpenCV 读取的图像
        :param detections: 检测结果列表
        :return: 带有检测框的图像
        """
        for det in detections:
            left, top, right, bottom = det["box"]

            # 绘制边界框
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            # 绘制类别和置信度
       # 在图像上添加标签，显示检测到的物体的类别和置信度
            label = f'{det["class_name"]}: {det["score"]:.2f}'
            # 在图像上添加文本，参数分别为：图像，文本内容，文本位置，字体，字体大小，字体颜色，字体粗细
            cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

