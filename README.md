The project is based on YOLO11 for building a model to detect objects such as doors, reflective vests, and five-point seat belts.

- `database` contains content related to database connections.
- `datasets` includes the results of dataset splitting.
- `detection` manages the logic for switching models between front-end and back-end.
- `runs` contains the results of each model run, including model files, model evaluation results, and reference outputs.
- `scripts` includes contents for training the five-point seat belt model.
- `static` holds simple front-end style files.
- `app.py` is the launch class for the visualization interface.

Note: Other `.py` files in the `yolo11_warning_sign` folder are mostly for training and testing models. In most cases, you don't need them; what's useful are the model files and the visualization interface. Be sure to change some of the paths in there.






本项目基于yolo11实现检测门、反光背心、五点式安全带等目标的模型构建


database里是数据库连接相关的内容

datasets是数据集划分之后的部分结果

detection是前后端切换模型的逻辑

runs是各模型运行的结果，包括模型文件、模型测评结果、模型输出的结果参考

scripts里是五点式安全带训练的内容

static里是简单的前端样式文件

app.py是可视化界面启动类

注：yolo11_warning_sign这个文件夹下的其他.py文件都是训练模型和测试模型的代码，绝大部分情况下你用不着
    对你来说，有用的无非是模型文件和可视化界面，里面有的一些路径记得改一下
