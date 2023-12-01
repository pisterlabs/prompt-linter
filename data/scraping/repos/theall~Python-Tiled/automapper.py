##
# automapper.py
# Copyright 2010-2012, Stefan Beller, stefanbeller@googlemail.com
#
# This file is part of Tiled.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.
##
from tilesetmanager import TilesetManager
from layer import Layer
from tilelayer import TileLayer
from objectgroup import ObjectGroup
from geometry import coherentRegions
from changeproperties import ChangeProperties
from automappingutils import objectsInRegion, tileRegionOfObjectGroup
from addremovetileset import AddTileset, RemoveTileset
from addremovemapobject import AddMapObject
from addremovelayer import AddLayer, RemoveLayer
from pyqtcore import (
    QVector,
    QMap,
    QSet,
    QString,
    QList,
    qrand
)
from PyQt5.QtCore import (
    Qt,
    QRect,
    QRectF,
    QObject,
    QVariant,
    QPoint
)
from PyQt5.QtGui import (
    QRegion
)
class InputIndexName():
    def __init__(self):
        self.listYes = QVector()
        self.listNo = QVector()

class InputIndex(QMap):
    def __init__(self):
        self.names = QSet()

class InputLayers(QMap):
    def __init__(self):
        self.indexes = QSet()
        self.names = QSet() # all names

class RuleOutput(QMap):
    def __init__(self):
        self.index = QString()

##
# This class does all the work for the automapping feature.
# basically it can do the following:
# - check the rules map for rules and store them
# - compare TileLayers (i. e. check if/where a certain rule must be applied)
# - copy regions of Maps (multiple Layers, the layerlist is a
#                         lookup-table for matching the Layers)
##
class AutoMapper(QObject):
    ##
    # Constructs an AutoMapper.
    # All data structures, which only rely on the rules map are setup
    # here.
    #
    # @param workingDocument: the map to work on.
    # @param rules: The rule map which should be used for automapping
    # @param rulePath: The filepath to the rule map.
    ##
    def __init__(self, workingDocument, rules, rulePath):
        ##
        # where to work in
        ##
        self.mMapDocument = workingDocument

        ##
        # the same as mMapDocument.map()
        ##
        self.mMapWork = None
        if workingDocument:
            self.mMapWork = workingDocument.map()

        ##
        # map containing the rules, usually different than mMapWork
        ##
        self.mMapRules = rules

        ##
        # This contains all added tilesets as pointers.
        # if rules use Tilesets which are not in the mMapWork they are added.
        # keep track of them, because we need to delete them afterwards,
        # when they still are unused
        # they will be added while setupTilesets().
        ##
        self.mAddedTilesets = QVector()

        ##
        # description see: mAddedTilesets, just described by Strings
        ##
        self.mAddedTileLayers = QList()

        ##
        # Points to the tilelayer, which defines the inputregions.
        ##
        self.mLayerInputRegions = None

        ##
        # Points to the tilelayer, which defines the outputregions.
        ##
        self.mLayerOutputRegions = None

        ##
        # Contains all tilelayer pointers, which names begin with input*
        # It is sorted by index and name
        ##
        self.mInputRules = InputLayers()

        ##
        # List of Regions in mMapRules to know where the input rules are
        ##
        self.mRulesInput = QList()

        ##
        # List of regions in mMapRules to know where the output of a
        # rule is.
        # mRulesOutput[i] is the output of that rule,
        # which has the input at mRulesInput[i], meaning that mRulesInput
        # and mRulesOutput must match with the indexes.
        ##
        self.mRulesOutput = QList()

        ##
        # The inner set with layers to indexes is needed for translating
        # tile layers from mMapRules to mMapWork.
        #
        # The key is the pointer to the layer in the rulemap. The
        # pointer to the layer within the working map is not hardwired, but the
        # position in the layerlist, where it was found the last time.
        # This loosely bound pointer ensures we will get the right layer, since we
        # need to check before anyway, and it is still fast.
        #
        # The list is used to hold different translation tables
        # => one of the tables is chosen by chance, so randomness is available
        ##
        self.mLayerList = QList()
        ##
        # store the name of the processed rules file, to have detailed
        # error messages available
        ##
        self.mRulePath = rulePath

        ##
        # determines if all tiles in all touched layers should be deleted first.
        ##
        self.mDeleteTiles = False

        ##
        # This variable determines, how many overlapping tiles should be used.
        # The bigger the more area is remapped at an automapping operation.
        # This can lead to higher latency, but provides a better behavior on
        # interactive automapping.
        # It defaults to zero.
        ##
        self.mAutoMappingRadius = 0

        ##
        # Determines if a rule is allowed to overlap it
        ##
        self.mNoOverlappingRules = False

        self.mTouchedObjectGroups = QSet()
        self.mWarning = QString()
        self.mTouchedTileLayers = QSet()
        self.mError = ''

        if (not self.setupRuleMapProperties()):
            return
        if (not self.setupRuleMapTileLayers()):
            return
        if (not self.setupRuleList()):
            return

    def __del__(self):
        self.cleanUpRulesMap()

    ##
    # Checks if the passed \a ruleLayerName is used in this instance
    # of Automapper.
    ##
    def ruleLayerNameUsed(self, ruleLayerName):
        return self.mInputRules.names.contains(ruleLayerName)

    ##
    # Call prepareLoad first! Returns a set of strings describing the tile
    # layers, which could be touched considering the given layers of the
    # rule map.
    ##
    def getTouchedTileLayers(self):
        return self.mTouchedTileLayers

    ##
    # This needs to be called directly before the autoMap call.
    # It sets up some data structures which change rapidly, so it is quite
    # painful to keep these datastructures up to date all time. (indices of
    # layers of the working map)
    ##
    def prepareAutoMap(self):
        self.mError = ''
        self.mWarning = ''
        if (not self.setupMissingLayers()):
            return False
        if (not self.setupCorrectIndexes()):
            return False
        if (not self.setupTilesets(self.mMapRules, self.mMapWork)):
            return False
        return True

    ##
    # Here is done all the automapping.
    ##
    def autoMap(self, where):
        # first resize the active area
        if (self.mAutoMappingRadius):
            region = QRegion()
            for r in where.rects():
                region += r.adjusted(- self.mAutoMappingRadius,
                                     - self.mAutoMappingRadius,
                                     + self.mAutoMappingRadius,
                                     + self.mAutoMappingRadius)

           #where += region

        # delete all the relevant area, if the property "DeleteTiles" is set
        if (self.mDeleteTiles):
            setLayersRegion = self.getSetLayersRegion()
            for i in range(self.mLayerList.size()):
                translationTable = self.mLayerList.at(i)
                for layer in translationTable.keys():
                    index = self.mLayerList.at(i).value(layer)
                    dstLayer = self.mMapWork.layerAt(index)
                    region = setLayersRegion.intersected(where)
                    dstTileLayer = dstLayer.asTileLayer()
                    if (dstTileLayer):
                        dstTileLayer.erase(region)
                    else:
                        self.eraseRegionObjectGroup(self.mMapDocument,
                                               dstLayer.asObjectGroup(),
                                               region)

        # Increase the given region where the next automapper should work.
        # This needs to be done, so you can rely on the order of the rules at all
        # locations
        ret = QRegion()
        for rect in where.rects():
            for i in range(self.mRulesInput.size()):
                # at the moment the parallel execution does not work yet
                # TODO: make multithreading available!
                # either by dividing the rules or the region to multiple threads
                ret = ret.united(self.applyRule(i, rect))

        #where = where.united(ret)

    ##
    # This cleans all datastructures, which are setup via prepareAutoMap,
    # so the auto mapper becomes ready for its next automatic mapping.
    ##
    def cleanAll(self):
        self.cleanTilesets()
        self.cleanTileLayers()

    ##
    # Contains all errors until operation was canceled.
    # The errorlist is cleared within prepareLoad and prepareAutoMap.
    ##
    def errorString(self):
        return self.mError

    ##
    # Contains all warnings which occur at loading a rules map or while
    # automapping.
    # The errorlist is cleared within prepareLoad and prepareAutoMap.
    ##
    def warningString(self):
        return self.mWarning

    ##
    # Reads the map properties of the rulesmap.
    # @return returns True when anything is ok, False when errors occured.
    ##
    def setupRuleMapProperties(self):
        properties = self.mMapRules.properties()
        for key in properties.keys():
            value = properties.value(key)
            raiseWarning = True
            if (key.toLower() == "deletetiles"):
                if (value.canConvert(QVariant.Bool)):
                    self.mDeleteTiles = value.toBool()
                    raiseWarning = False
            elif (key.toLower() == "automappingradius"):
                if (value.canConvert(QVariant.Int)):
                    self.mAutoMappingRadius = value
                    raiseWarning = False
            elif (key.toLower() == "nooverlappingrules"):
                if (value.canConvert(QVariant.Bool)):
                    self.mNoOverlappingRules = value.toBool()
                    raiseWarning = False

            if (raiseWarning):
                self.mWarning += self.tr("'%s': Property '%s' = '%s' does not make sense. \nIgnoring this property."%(self.mRulePath, key, value.toString()) + '\n')

        return True

    def cleanUpRulesMap(self):
        self.cleanTilesets()
        # mMapRules can be empty, when in prepareLoad the very first stages fail.
        if (not self.mMapRules):
            return
        tilesetManager = TilesetManager.instance()
        tilesetManager.removeReferences(self.mMapRules.tilesets())
        del self.mMapRules
        self.mMapRules = None
        self.cleanUpRuleMapLayers()
        self.mRulesInput.clear()
        self.mRulesOutput.clear()

    ##
    # Searches the rules layer for regions and stores these in \a rules.
    # @return returns True when anything is ok, False when errors occured.
    ##
    def setupRuleList(self):
        combinedRegions = coherentRegions(
                self.mLayerInputRegions.region() +
                self.mLayerOutputRegions.region())
        combinedRegions = QList(sorted(combinedRegions, key=lambda x:x.y(), reverse=True))
        rulesInput = coherentRegions(
                self.mLayerInputRegions.region())
        rulesOutput = coherentRegions(
                self.mLayerOutputRegions.region())
        for i in range(combinedRegions.size()):
            self.mRulesInput.append(QRegion())
            self.mRulesOutput.append(QRegion())

        for reg in rulesInput:
            for i in range(combinedRegions.size()):
                if (reg.intersects(combinedRegions[i])):
                    self.mRulesInput[i] += reg
                    break

        for reg in rulesOutput:
            for i in range(combinedRegions.size()):
                if (reg.intersects(combinedRegions[i])):
                    self.mRulesOutput[i] += reg
                    break

        for i in range(self.mRulesInput.size()):
            checkCoherent = self.mRulesInput.at(i).united(self.mRulesOutput.at(i))
            coherentRegions(checkCoherent).length() == 1
        return True

    ##
    # Sets up the layers in the rules map, which are used for automapping.
    # The layers are detected and put in the internal data structures
    # @return returns True when anything is ok, False when errors occured.
    ##
    def setupRuleMapTileLayers(self):
        error = QString()
        for layer in self.mMapRules.layers():
            layerName = layer.name()
            if (layerName.lower().startswith("regions")):
                treatAsBoth = layerName.toLower() == "regions"
                if (layerName.lower().endswith("input") or treatAsBoth):
                    if (self.mLayerInputRegions):
                        error += self.tr("'regions_input' layer must not occur more than once.\n")

                    if (layer.isTileLayer()):
                        self.mLayerInputRegions = layer.asTileLayer()
                    else:
                        error += self.tr("'regions_*' layers must be tile layers.\n")

                if (layerName.lower().endswith("output") or treatAsBoth):
                    if (self.mLayerOutputRegions):
                        error += self.tr("'regions_output' layer must not occur more than once.\n")

                    if (layer.isTileLayer()):
                        self.mLayerOutputRegions = layer.asTileLayer()
                    else:
                        error += self.tr("'regions_*' layers must be tile layers.\n")

                continue

            nameStartPosition = layerName.indexOf('_') + 1
            # name is all characters behind the underscore (excluded)
            name = layerName.right(layerName.size() - nameStartPosition)
            # group is all before the underscore (included)
            index = layerName.left(nameStartPosition)
            if (index.lower().startswith("output")):
                index.remove(0, 6)
            elif (index.lower().startswith("inputnot")):
                index.remove(0, 8)
            elif (index.lower().startswith("input")):
                index.remove(0, 5)
            # both 'rule' and 'output' layers will require and underscore and 
            # rely on the correct position detected of the underscore
            if (nameStartPosition == 0):
                error += self.tr("Did you forget an underscore in layer '%d'?\n"%layerName)
                continue

            if (layerName.startsWith("input", Qt.CaseInsensitive)):
                isNotList = layerName.lower().startswith("inputnot")
                if (not layer.isTileLayer()):
                    error += self.tr("'input_*' and 'inputnot_*' layers must be tile layers.\n")
                    continue

                self.mInputRules.names.insert(name)
                if (not self.mInputRules.indexes.contains(index)):
                    self.mInputRules.indexes.insert(index)
                    self.mInputRules.insert(index, InputIndex())

                if (not self.mInputRules[index].names.contains(name)):
                    self.mInputRules[index].names.insert(name)
                    self.mInputRules[index].insert(name, InputIndexName())

                if (isNotList):
                    self.mInputRules[index][name].listNo.append(layer.asTileLayer())
                else:
                    self.mInputRules[index][name].listYes.append(layer.asTileLayer())
                continue

            if layerName.lower().startswith("output"):
                if (layer.isTileLayer()):
                    self.mTouchedTileLayers.insert(name)
                else:
                    self.mTouchedObjectGroups.insert(name)
                type = layer.layerType()
                layerIndex = self.mMapWork.indexOfLayer(name, type)
                found = False
                for translationTable in self.mLayerList:
                    if (translationTable.index == index):
                        translationTable.insert(layer, layerIndex)
                        found = True
                        break

                if (not found):
                    self.mLayerList.append(RuleOutput())
                    self.mLayerList.last().insert(layer, layerIndex)
                    self.mLayerList.last().index = index

                continue

            error += self.tr("Layer '%s' is not recognized as a valid layer for Automapping.\n"%layerName)

        if (not self.mLayerInputRegions):
            error += self.tr("No 'regions' or 'regions_input' layer found.\n")
        if (not self.mLayerOutputRegions):
            error += self.tr("No 'regions' or 'regions_output' layer found.\n")
        if (self.mInputRules.isEmpty()):
            error += self.tr("No input_<name> layer found!\n")
        # no need to check for mInputNotRules.size() == 0 here.
        # these layers are not necessary.
        if error != '':
            error = self.mRulePath + '\n' + error
            self.mError += error
            return False

        return True

    ##
    # Checks if all needed layers in the working map are there.
    # If not, add them in the correct order.
    ##
    def setupMissingLayers(self):
        # make sure all needed layers are there:
        for name in self.mTouchedTileLayers:
            if (self.mMapWork.indexOfLayer(name, Layer.TileLayerType) != -1):
                continue
            index = self.mMapWork.layerCount()
            tilelayer = TileLayer(name, 0, 0, self.mMapWork.width(), self.mMapWork.height())
            self.mMapDocument.undoStack().push(AddLayer(self.mMapDocument, index, tilelayer))
            self.mAddedTileLayers.append(name)

        for name in self.mTouchedObjectGroups:
            if (self.mMapWork.indexOfLayer(name, Layer.ObjectGroupType) != -1):
                continue
            index = self.mMapWork.layerCount()
            objectGroup = ObjectGroup(name, 0, 0,
                                                       self.mMapWork.width(),
                                                       self.mMapWork.height())
            self.mMapDocument.undoStack().push(AddLayer(self.mMapDocument, index, objectGroup))
            self.mAddedTileLayers.append(name)

        return True

    ##
    # Checks if the layers setup as in setupRuleMapLayers are still right.
    # If it's not right, correct them.
    # @return returns True if everything went fine. False is returned when
    #         no set layer was found
    ##
    def setupCorrectIndexes(self):
        # make sure all indexes of the layer translationtables are correct.
        for i in range(self.mLayerList.size()):
            translationTable = self.mLayerList.at(i)
            for layerKey in translationTable.keys():
                name = layerKey.name()
                pos = name.indexOf('_') + 1
                name = name.right(name.length() - pos)
                index = translationTable.value(layerKey, -1)
                if (index >= self.mMapWork.layerCount() or index == -1 or
                        name != self.mMapWork.layerAt(index).name()):
                    newIndex = self.mMapWork.indexOfLayer(name, layerKey.layerType())
                    translationTable.insert(layerKey, newIndex)

        return True

    ##
    # sets up the tilesets which are used in automapping.
    # @return returns True when anything is ok, False when errors occured.
    #        (in that case will be a msg box anyway)
    ##
    # This cannot just be replaced by MapDocument::unifyTileset(Map),
    # because here mAddedTileset is modified.
    def setupTilesets(self, src, dst):
        existingTilesets = dst.tilesets()
        tilesetManager = TilesetManager.instance()
        # Add tilesets that are not yet part of dst map
        for tileset in src.tilesets():
            if (existingTilesets.contains(tileset)):
                continue
            undoStack = self.mMapDocument.undoStack()
            replacement = tileset.findSimilarTileset(existingTilesets)
            if (not replacement):
                self.mAddedTilesets.append(tileset)
                undoStack.push(AddTileset(self.mMapDocument, tileset))
                continue

            # Merge the tile properties
            sharedTileCount = min(tileset.tileCount(), replacement.tileCount())
            for i in range(sharedTileCount):
                replacementTile = replacement.tileAt(i)
                properties = replacementTile.properties()
                properties.merge(tileset.tileAt(i).properties())
                undoStack.push(ChangeProperties(self.mMapDocument,
                                                     self.tr("Tile"),
                                                     replacementTile,
                                                     properties))

            src.replaceTileset(tileset, replacement)
            tilesetManager.addReference(replacement)
            tilesetManager.removeReference(tileset)

        return True

    ##
    # Returns the conjunction of of all regions of all setlayers
    ##
    def getSetLayersRegion(self):
        result = QRegion()
        for name in self.mInputRules.names:
            index = self.mMapWork.indexOfLayer(name, Layer.TileLayerType)
            if (index == -1):
                continue
            setLayer = self.mMapWork.layerAt(index).asTileLayer()
            result |= setLayer.region()

        return result

    ##
    # This copies all Tiles from TileLayer src to TileLayer dst
    #
    # In src the Tiles are taken from the rectangle given by
    # src_x, src_y, width and height.
    # In dst they get copied to a rectangle given by
    # dst_x, dst_y, width, height .
    # if there is no tile in src TileLayer, there will nothing be copied,
    # so the maybe existing tile in dst will not be overwritten.
    #
    ##
    def copyTileRegion(self, srcLayer, srcX, srcY, width, height, dstLayer, dstX, dstY):
        startX = max(dstX, 0)
        startY = max(dstY, 0)
        endX = min(dstX + width, dstLayer.width())
        endY = min(dstY + height, dstLayer.height())
        offsetX = srcX - dstX
        offsetY = srcY - dstY
        for x in range(startX, endX):
            for y in range(startY, endY):
                cell = srcLayer.cellAt(x + offsetX, y + offsetY)
                if (not cell.isEmpty()):
                    # this is without graphics update, it's done afterwards for all
                    dstLayer.setCell(x, y, cell)

    ##
    # This copies all objects from the \a src_lr ObjectGroup to the \a dst_lr
    # in the given rectangle.
    #
    # The rectangle is described by the upper left corner \a src_x \a src_y
    # and its \a width and \a height. The parameter \a dst_x and \a dst_y
    # offset the copied objects in the destination object group.
    ##
    def copyObjectRegion(self, srcLayer, srcX, srcY, width, height, dstLayer, dstX, dstY):
        undo = self.mMapDocument.undoStack()
        rect = QRectF(srcX, srcY, width, height)
        pixelRect = self.mMapDocument.renderer().tileToPixelCoords_(rect)
        objects = objectsInRegion(srcLayer, pixelRect.toAlignedRect())
        pixelOffset = self.mMapDocument.renderer().tileToPixelCoords(dstX, dstY)
        pixelOffset -= pixelRect.topLeft()
        clones = QList()
        for obj in objects:
            clone = obj.clone()
            clones.append(clone)
            clone.setX(clone.x() + pixelOffset.x())
            clone.setY(clone.y() + pixelOffset.y())
            undo.push(AddMapObject(self.mMapDocument, dstLayer, clone))

    ##
    # This copies multiple TileLayers from one map to another.
    # Only the region \a region is considered for copying.
    # In the destination it will come to the region translated by Offset.
    # The parameter \a LayerTranslation is a map of which layers of the rulesmap
    # should get copied into which layers of the working map.
    ##
    def copyMapRegion(self, region, offset, layerTranslation):
        for i in range(layerTranslation.keys().size()):
            _from = layerTranslation.keys().at(i)
            to = self.mMapWork.layerAt(layerTranslation.value(_from))
            for rect in region.rects():
                fromTileLayer = _from.asTileLayer()
                fromObjectGroup = _from.asObjectGroup()
                if (fromTileLayer):
                    toTileLayer = to.asTileLayer()
                    self.copyTileRegion(fromTileLayer, rect.x(), rect.y(),
                                   rect.width(), rect.height(),
                                   toTileLayer,
                                   rect.x() + offset.x(), rect.y() + offset.y())
                elif (fromObjectGroup):
                    toObjectGroup = to.asObjectGroup()
                    self.copyObjectRegion(fromObjectGroup, rect.x(), rect.y(),
                                     rect.width(), rect.height(),
                                     toObjectGroup,
                                     rect.x() + offset.x(), rect.y() + offset.y())
                else:
                    pass

    ##
    # This goes through all the positions of the mMapWork and checks if
    # there fits the rule given by the region in mMapRuleSet.
    # if there is a match all Layers are copied to mMapWork.
    # @param ruleIndex: the region which should be compared to all positions
    #              of mMapWork will be looked up in mRulesInput and mRulesOutput
    # @return where: an rectangle where the rule actually got applied
    ##
    def applyRule(self, ruleIndex, where):
        ret = QRect()
        if (self.mLayerList.isEmpty()):
            return ret
        ruleInput = self.mRulesInput.at(ruleIndex)
        ruleOutput = self.mRulesOutput.at(ruleIndex)
        rbr = ruleInput.boundingRect()
        # Since the rule itself is translated, we need to adjust the borders of the
        # loops. Decrease the size at all sides by one: There must be at least one
        # tile overlap to the rule.
        minX = where.left() - rbr.left() - rbr.width() + 1
        minY = where.top() - rbr.top() - rbr.height() + 1
        maxX = where.right() - rbr.left() + rbr.width() - 1
        maxY = where.bottom() - rbr.top() + rbr.height() - 1
        # In this list of regions it is stored which parts or the map have already
        # been altered by exactly this rule. We store all the altered parts to
        # make sure there are no overlaps of the same rule applied to
        # (neighbouring) places
        appliedRegions = QList()
        if (self.mNoOverlappingRules):
            for i in range(self.mMapWork.layerCount()):
                appliedRegions.append(QRegion())
        for y in range(minY, maxY+1):
            for x in range(minX, maxX+1):
                anymatch = False
                for index in self.mInputRules.indexes:
                    ii = self.mInputRules[index]
                    allLayerNamesMatch = True
                    for name in ii.names:
                        i = self.mMapWork.indexOfLayer(name, Layer.TileLayerType)
                        if (i == -1):
                            allLayerNamesMatch = False
                        else:
                            setLayer = self.mMapWork.layerAt(i).asTileLayer()
                            allLayerNamesMatch &= compareLayerTo(setLayer,
                                                                 ii[name].listYes,
                                                                 ii[name].listNo,
                                                                 ruleInput,
                                                                 QPoint(x, y))

                    if (allLayerNamesMatch):
                        anymatch = True
                        break

                if (anymatch):
                    r = 0
                    # choose by chance which group of rule_layers should be used:
                    if (self.mLayerList.size() > 1):
                        r = qrand() % self.mLayerList.size()
                    if (not self.mNoOverlappingRules):
                        self.copyMapRegion(ruleOutput, QPoint(x, y), self.mLayerList.at(r))
                        ret = ret.united(rbr.translated(QPoint(x, y)))
                        continue

                    missmatch = False
                    translationTable = self.mLayerList.at(r)
                    layers = translationTable.keys()
                    # check if there are no overlaps within this rule.
                    ruleRegionInLayer = QVector()
                    for i in range(layers.size()):
                        layer = layers.at(i)
                        appliedPlace = QRegion()
                        tileLayer = layer.asTileLayer()
                        if (tileLayer):
                            appliedPlace = tileLayer.region()
                        else:
                            appliedPlace = tileRegionOfObjectGroup(layer.asObjectGroup())
                        ruleRegionInLayer.append(appliedPlace.intersected(ruleOutput))
                        if (appliedRegions.at(i).intersects(
                                    ruleRegionInLayer[i].translated(x, y))):
                            missmatch = True
                            break

                    if (missmatch):
                        continue
                    self.copyMapRegion(ruleOutput, QPoint(x, y), self.mLayerList.at(r))
                    ret = ret.united(rbr.translated(QPoint(x, y)))
                    for i in range(translationTable.size()):
                        appliedRegions[i] += ruleRegionInLayer[i].translated(x, y)

        return ret

    ##
    # Cleans up the data structes filled by setupRuleMapLayers(),
    # so the next rule can be processed.
    ##
    def cleanUpRuleMapLayers(self):
        self.cleanTileLayers()
        it = QList.const_iterator()
        for it in self.mLayerList:
            del it
        self.mLayerList.clear()
        # do not delete mLayerRuleRegions, it is owned by the rulesmap
        self.mLayerInputRegions = None
        self.mLayerOutputRegions = None
        self.mInputRules.clear()

    ##
    # Cleans up the data structes filled by setupTilesets(),
    # so the next rule can be processed.
    ##
    def cleanTilesets(self):
        for tileset in self.mAddedTilesets:
            if (self.mMapWork.isTilesetUsed(tileset)):
                continue
            index = self.mMapWork.indexOfTileset(tileset)
            if (index == -1):
                continue
            undo = self.mMapDocument.undoStack()
            undo.push(RemoveTileset(self.mMapDocument, index))

        self.mAddedTilesets.clear()

    ##
    # Cleans up the added tile layers setup by setupMissingLayers(),
    # so we have a minimal addition of tile layers by the automapping.
    ##
    def cleanTileLayers(self):
        for tilelayerName in self.mAddedTileLayers:
            layerIndex = self.mMapWork.indexOfLayer(tilelayerName,
                                                          Layer.TileLayerType)
            if (layerIndex == -1):
                continue
            layer = self.mMapWork.layerAt(layerIndex)
            if (not layer.isEmpty()):
                continue
            undo = self.mMapDocument.undoStack()
            undo.push(RemoveLayer(self.mMapDocument, layerIndex))

        self.mAddedTileLayers.clear()

##
# About the order of the methods in this file.
# The Automapper class has 3 bigger public functions, that is
# prepareLoad(), prepareAutoMap() and autoMap().
# These three functions make use of lots of different private methods, which
# are put directly below each of these functions.
##
def compareRuleRegion(r1, r2):
    p1 = r1.boundingRect().topLeft()
    p2 = r2.boundingRect().topLeft()
    return p1.y() < p2.y() or (p1.y() == p2.y() and p1.x() < p2.x())

##
# Returns a list of all cells which can be found within all tile layers
# within the given region.
##
def cellsInRegion(list, r):
    cells = QVector()
    for tilelayer in list:
        for rect in r.rects():
            for x in range(rect.left(), rect.right()+1):
                for y in range(rect.top(), rect.bottom()+1):
                    cell = tilelayer.cellAt(x, y)
                    if (not cells.contains(cell)):
                        cells.append(cell)

    return cells

##
# This function is one of the core functions for understanding the
# automapping.
# In this function a certain region (of the set layer) is compared to
# several other layers (ruleSet and ruleNotSet).
# This comparision will determine if a rule of automapping matches,
# so if this rule is applied at this region given
# by a QRegion and Offset given by a QPoint.
#
# This compares the tile layer setLayer to several others given
# in the QList listYes (ruleSet) and OList listNo (ruleNotSet).
# The tile layer setLayer is examined at QRegion ruleRegion + offset
# The tile layers within listYes and listNo are examined at QRegion ruleRegion.
#
# Basically all matches between setLayer and a layer of listYes are considered
# good, while all matches between setLayer and listNo are considered bad and
# lead to canceling the comparison, returning False.
#
# The comparison is done for each position within the QRegion ruleRegion.
# If all positions of the region are considered "good" return True.
#
# Now there are several cases to distinguish:
#  - both listYes and listNo are empty:
#      This should not happen, because with that configuration, absolutely
#      no condition is given.
#      return False, assuming this is an errornous rule being applied
#
#  - both listYes and listNo are not empty:
#      When comparing a tile at a certain position of tile layer setLayer
#      to all available tiles in listYes, there must be at least
#      one layer, in which there is a match of tiles of setLayer and
#      listYes to consider this position good.
#      In listNo there must not be a match to consider this position
#      good.
#      If there are no tiles within all available tiles within all layers
#      of one list, all tiles in setLayer are considered good,
#      while inspecting this list.
#      All available tiles are all tiles within the whole rule region in
#      all tile layers of the list.
#
#  - either of both lists are not empty
#      When comparing a certain position of tile layer setLayer
#      to all Tiles at the corresponding position this can happen:
#      A tile of setLayer matches a tile of a layer in the list. Then this
#      is considered as good, if the layer is from the listYes.
#      Otherwise it is considered bad.
#
#      Exception, when having only the listYes:
#      if at the examined position there are no tiles within all Layers
#      of the listYes, all tiles except all used tiles within
#      the layers of that list are considered good.
#
#      This exception was added to have a better functionality
#      (need of less layers.)
#      It was not added to the case, when having only listNo layers to
#      avoid total symmetry between those lists.
#
# If all positions are considered good, return True.
# return False otherwise.
#
# @return bool, if the tile layer matches the given list of layers.
##
def compareLayerTo(setLayer, listYes, listNo, ruleRegion, offset):
    if (listYes.isEmpty() and listNo.isEmpty()):
        return False
    cells = QVector()
    if (listYes.isEmpty()):
        cells = cellsInRegion(listNo, ruleRegion)
    if (listNo.isEmpty()):
        cells = cellsInRegion(listYes, ruleRegion)
    for rect in ruleRegion.rects():
        for x in range(rect.left(), rect.right()+1):
            for y in range(rect.top(), rect.bottom()+1):
                # this is only used in the case where only one list has layers
                # it is needed for the exception mentioned above
                ruleDefinedListYes = False
                matchListYes = False
                matchListNo  = False
                if (not setLayer.contains(x + offset.x(), y + offset.y())):
                    return False
                c1 = setLayer.cellAt(x + offset.x(),
                                                  y + offset.y())
                # ruleDefined will be set when there is a tile in at least
                # one layer. if there is a tile in at least one layer, only
                # the given tiles in the different listYes layers are valid.
                # if there is given no tile at all in the listYes layers,
                # consider all tiles valid.
                for comparedTileLayer in listYes:
                    if (not comparedTileLayer.contains(x, y)):
                        return False
                    c2 = comparedTileLayer.cellAt(x, y)
                    if (not c2.isEmpty()):
                        ruleDefinedListYes = True
                    if (not c2.isEmpty() and c1 == c2):
                        matchListYes = True

                for comparedTileLayer in listNo:
                    if (not comparedTileLayer.contains(x, y)):
                        return False
                    c2 = comparedTileLayer.cellAt(x, y)
                    if (not c2.isEmpty() and c1 == c2):
                        matchListNo = True

                # when there are only layers in the listNo
                # check only if these layers are unmatched
                # no need to check explicitly the exception in this case.
                if (listYes.isEmpty()):
                    if (matchListNo):
                        return False
                    else:
                        continue

                # when there are only layers in the listYes
                # check if these layers are matched, or if the exception works
                if (listNo.isEmpty()):
                    if (matchListYes):
                        continue
                    if (not ruleDefinedListYes and not cells.contains(c1)):
                        continue
                    return False

                # there are layers in both lists:
                # no need to consider ruleDefinedListXXX
                if ((matchListYes or not ruleDefinedListYes) and not matchListNo):
                    continue
                else:
                    return False

    return True
