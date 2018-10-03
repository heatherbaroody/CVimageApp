import kivy
kivy.require('1.9.0')
import LearningAlgorithm

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty



class CVtest(BoxLayout):
    def capture(self):
        camera = self.ids['camera']
        camera.export_to_png("IMG_.png")
        print('captured')

    CVresult = StringProperty()
    KNNresult = StringProperty()
    KNNfacetnames = StringProperty()
    # capture image from camera
    
    #process image to readable form and  #run CVanalyze function in learning algorithm
    def analyzeButton(self,capturedImage):
        result = LearningAlgorithm.analyzeCV(capturedImage)
        self.CVresult = result
        return result
    
    def KNNanalyzeButton(self, KNNcapturedImage):
        [KNN_facet, KNN_prob]= LearningAlgorithm.KNNpredict(KNNcapturedImage)
        KNN_prob = str(KNN_prob)
        self.KNNresult = KNN_prob
        self.KNNfacetnames = '[Pt (111), Pt (110), Pt (100), Pt(poly)]'
        #print (KNN_prob, KNN_prob)
        return [KNN_prob, KNN_prob]


class CVtestApp(App):
    def build(self):
        return CVtest()

cvApp = CVtestApp()
cvApp.run()
    
    
    
    
    
    
    
    
#    def build(self):
#       return BoxLayout()

#img = Image.open("pt100-test.jpg").convert('I')
#LearningAlgorithm.analyzeCV(img)

#flApp = CVtestApp()

#flApp.run()